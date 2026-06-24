"""
Knowledge Graph builder: constructs daily KG snapshots from Aeolus_V2 tri-modal data.

Architecture (Single Shared GAT):
  - Main KG graph: Flight + Airport + Airline + Aircraft nodes (8 relation types)
    * Node features: 25-dim (20 flight + 5 airport)
  - Chain graph: Flight→Flight (same aircraft, preceded_by)
    * Node features: 25-dim (shared with Main KG features)
    * Edge features: 3-dim [gap_time, dep_delay_A, arr_delay_A]
  - Network heterogeneous graph: Airport + Flight joint nodes (3 edge types)
    * Node features: 25-dim (20 flight + 5 airport)
    * Edge type 0: Airport→Airport (route busy level)
    * Edge type 1: Flight→Airport (departure relation)
    * Edge type 2: Airport→Flight (arrival relation)

Hypothesis: Performance differences among three GATs arise from different graph structures, not different parameters.
  → Use a single shared GAT to process all three graphs and test AUC changes.
"""
import torch
import dgl
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.config import CONFIG


class TimeEncoder(torch.nn.Module):
    """Sinusoidal time encoding for daily snapshots."""

    def __init__(self, dim: int = 64, max_period: int = 365):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        # Precompute sinusoidal encoding
        pe = torch.zeros(max_period, dim)
        position = torch.arange(0, max_period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() *
                             (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, day_of_year: int) -> torch.Tensor:
        """Return time encoding for a given day (0-indexed)."""
        return self.pe[day_of_year % self.max_period]  # (dim,)


class DailyKGBuilder:
    """Builds a daily KG snapshot from tabular + chain + network data."""

    def __init__(self):
        self.cfg = CONFIG.kg
        self.time_encoder = TimeEncoder(
            dim=self.cfg.time_encoding_dim,
            max_period=self.cfg.time_encoding_max_period
        )
        # Entity ID mappings (per snapshot)
        self.entity_ids = {}
        self.next_entity_id = 0
        
        # Feature normalization (P0-1)
        self.feat_scaler = None
        self.feat_sum = None
        self.feat_sq_sum = None
        self.feat_count = 0
        
        # Label encoders for categorical features (P0-2)
        self.label_encoders = {}
        self.encoders_fitted = False

    def _get_or_create_id(self, key: str) -> int:
        if key not in self.entity_ids:
            self.entity_ids[key] = self.next_entity_id
            self.next_entity_id += 1
        return self.entity_ids[key]

    def build(self, year: int, month: int, day: int,
              tabular_df, chain_data=None, network_graph=None):
        """
        Build a single KG snapshot for a given day.
        Uses a structured node layout:
        - Flight nodes: IDs [0, n_flights-1] - aligned with tabular row order
        - Other entities: IDs [n_flights, ...]

        This ensures Flight node i corresponds to tabular row i.

        Args:
            tabular_df: DataFrame with flight records (one row per flight)
            chain_data: dict from .pt chain file (unused, kept for API compatibility)
            network_graph: DGLGraph from .dgl network file (unused, kept for API compatibility)

        Returns:
            tuple: (g, time_enc, n_flights, g_chain, g_network, airport_flight_map)
              - g: main KG DGLGraph
              - time_enc: (64,) time encoding tensor
              - n_flights: number of flight nodes
              - g_chain: Chain graph (Flight→Flight, preceded_by)
              - g_network: Network heterogeneous graph (Airport+Flight joint, 3 edge types)
                  * Nodes: Airport (6-dim features) + Flight (10-dim features)
                  * Edge types: 0=Airport→Airport, 1=Flight→Airport, 2=Airport→Flight
              - airport_flight_map: dict {
                  'flight_node_offset': int, starting ID of Flight nodes
                  'origin_ap_ids': LongTensor (n_flights,), origin airport node ID per flight
                  'dest_ap_ids': LongTensor (n_flights,), destination airport node ID per flight
              }
        """
        self.entity_ids.clear()
        self.next_entity_id = 0
        day_of_year = self._day_of_year(year, month, day)
        time_enc = self.time_encoder(day_of_year)  # (64,)

        # Count flights first to reserve ID space
        n_flights = len(tabular_df)
        
        # Reserve ID space for flights: they get IDs 0 to n_flights-1
        # Other entities will start from n_flights
        flight_ids = list(range(n_flights))
        self.next_entity_id = n_flights  # Non-flight entities start here

        # Compute day-of-year
        edges = {rel: {"src": [], "dst": [], "feat": []}
                 for rel in CONFIG.kg.relation_types}

        # Entity features placeholder
        entity_feats = {}

        # P2-8: Airport traffic statistics
        airport_flight_count = {}
        for _, row in tabular_df.iterrows():
            origin_o = str(row.get("ORIGIN_INDEX", ""))
            dest_o = str(row.get("DEST_INDEX", ""))
            airport_flight_count[origin_o] = airport_flight_count.get(origin_o, 0) + 1
            airport_flight_count[dest_o] = airport_flight_count.get(dest_o, 0) + 1
        
        # Additional features: O_DAY_FLIGHTS, D_DAY_FLIGHTS, IS_PEAK_HOUR
        tabular_df = tabular_df.copy()
        tabular_df["O_DAY_FLIGHTS"] = tabular_df["ORIGIN_INDEX"].apply(
            lambda x: airport_flight_count.get(str(x), 0))
        tabular_df["D_DAY_FLIGHTS"] = tabular_df["DEST_INDEX"].apply(
            lambda x: airport_flight_count.get(str(x), 0))
        tabular_df["IS_PEAK_HOUR"] = tabular_df["CRS_DEP_TIME_MIN"].apply(
            lambda t: 1.0 if 7*60 <= float(t) <= 9*60 or 17*60 <= float(t) <= 19*60 else 0.0)

        # Unified 25-dim features for all three modalities (20 flight + 5 airport), shared GAT
        ap_to_idx, n_airports_main, ap_feat_5d = self._compute_airport_features(tabular_df)

        # --- Process each flight ---
        for row_idx, (_, row) in enumerate(tabular_df.iterrows()):
            tail_num = str(row.get("TAIL_NUM", ""))
            carrier = str(row.get("OP_CARRIER", ""))
            origin = str(row.get("ORIGIN_INDEX", ""))
            dest = str(row.get("DEST_INDEX", ""))
            fl_num = str(row.get("OP_CARRIER_FL_NUM", ""))
            crs_dep = float(row.get("CRS_DEP_TIME_MIN", 0))
            
            origin_tier = int(row.get("ORIGIN_TIER", 0))
            dest_tier = int(row.get("DEST_TIER", 0))
            prev_dep_delay = float(row.get("PREV_DEP_DELAY", 0.0))
            prev_arr_delay = float(row.get("PREV_ARR_DELAY", 0.0))

            o_wspd = float(row.get("O_WSPD", 0))
            o_prcp = float(row.get("O_PRCP", 0))
            if o_wspd > 25 or o_prcp > 10:
                weather_type = 0
            elif o_wspd > 15 or o_prcp > 5:
                weather_type = 1
            else:
                weather_type = 2

            flight_id = flight_ids[row_idx]
            aircraft_id = self._get_or_create_id(f"A:{tail_num}")
            origin_airport_id = self._get_or_create_id(f"AP:{origin}")
            dest_airport_id = self._get_or_create_id(f"AP:{dest}")
            airline_id = self._get_or_create_id(f"AL:{carrier}")

            # Unified 25-dim features (20 flight + 5 airport) — shared across modalities
            node_feat = self._build_flight_node_feat(row)
            entity_feats[flight_id] = torch.cat([node_feat, torch.zeros(5)])

            # Aircraft node (25-dim)
            entity_feats.setdefault(aircraft_id, torch.zeros(25))

            # Airport node (25-dim: first 20=0 + last 5=airport features)
            if origin in ap_to_idx:
                ap_5d = torch.tensor(ap_feat_5d[ap_to_idx[origin]], dtype=torch.float32)
                entity_feats[origin_airport_id] = torch.cat([torch.zeros(20), ap_5d])
            if dest in ap_to_idx:
                ap_5d = torch.tensor(ap_feat_5d[ap_to_idx[dest]], dtype=torch.float32)
                entity_feats[dest_airport_id] = torch.cat([torch.zeros(20), ap_5d])

            # Relations
            # departs_from: edge_feat encodes predecessor delay intensity
            chain_strength = min(abs(prev_dep_delay) / 60.0, 1.0)
            edges["departs_from"]["src"].append(flight_id)
            edges["departs_from"]["dst"].append(origin_airport_id)
            edges["departs_from"]["feat"].append(chain_strength)

            # arrives_at
            edges["arrives_at"]["src"].append(flight_id)
            edges["arrives_at"]["dst"].append(dest_airport_id)
            edges["arrives_at"]["feat"].append(0.0)

            # operated_by
            edges["operated_by"]["src"].append(flight_id)
            edges["operated_by"]["dst"].append(airline_id)
            edges["operated_by"]["feat"].append(0.0)

            # flown_by
            edges["flown_by"]["src"].append(flight_id)
            edges["flown_by"]["dst"].append(aircraft_id)
            edges["flown_by"]["feat"].append(0.0)

        # --- Chain data: preceded_by edges (independent subgraph, edge features carry predecessor actual delays) ---
        # Predecessor flights have already flown; their DEP_DELAY/ARR_DELAY are known facts (causal features, not label leakage)
        chain_src, chain_dst, chain_feat_val = [], [], []
        tabular_tmp = tabular_df.reset_index(drop=True).copy()
        tabular_tmp['_dep'] = pd.to_numeric(
            tabular_tmp['CRS_DEP_TIME_MIN'], errors='coerce').fillna(0)
        tabular_tmp['_arr'] = pd.to_numeric(
            tabular_tmp['CRS_ARR_TIME_MIN'], errors='coerce').fillna(0)
        for tail, grp in tabular_tmp.groupby('TAIL_NUM', sort=False):
            grp = grp.sort_values('_dep')
            idxs = grp.index.values
            for i in range(1, len(idxs)):
                prev_idx = int(idxs[i-1])       # A (predecessor flight)
                curr_idx = int(idxs[i])          # B (current flight)
                gap = float(grp.iloc[i]['_dep'] - grp.iloc[i-1]['_arr'])
                dep_delay_a = float(grp.iloc[i-1].get('DEP_DELAY', 0))
                arr_delay_a = float(grp.iloc[i-1].get('ARR_DELAY', 0))
                
                # Fix 1: NaN → 0.0 (treated as no delay)
                dep_delay_a = np.nan_to_num(dep_delay_a, nan=0.0)
                arr_delay_a = np.nan_to_num(arr_delay_a, nan=0.0)
                
                # Fix 2: Preserve positive/negative semantics (positive = delay, negative = early)
                dep_norm = max(min(dep_delay_a / 120.0, 1.0), -1.0)
                arr_norm = max(min(arr_delay_a / 120.0, 1.0), -1.0)
                
                chain_src.append(prev_idx)
                chain_dst.append(curr_idx)
                chain_feat_val.append([max(gap, 0.0), dep_norm, arr_norm])

        # --- Build homogeneous graph with etype edge data ---
        all_src = []
        all_dst = []
        all_etype = []
        all_feat = []
        
        relation_map = {rel: i for i, rel in enumerate(CONFIG.kg.relation_types)}
        
        for rel_name, rel_data in edges.items():
            if len(rel_data["src"]) > 0:
                all_src.extend(rel_data["src"])
                all_dst.extend(rel_data["dst"])
                rel_id = relation_map.get(rel_name, 0)
                all_etype.extend([rel_id] * len(rel_data["src"]))
                all_feat.extend(rel_data["feat"])

        if len(all_src) > 0:
            src_t = torch.tensor(all_src, dtype=torch.int64)
            dst_t = torch.tensor(all_dst, dtype=torch.int64)
            etype_t = torch.tensor(all_etype, dtype=torch.int64)
            feat_t = torch.tensor(all_feat, dtype=torch.float32).unsqueeze(-1)
            
            g = dgl.graph((src_t, dst_t), num_nodes=self.next_entity_id)
            g.edata[dgl.ETYPE] = etype_t
            g.edata["feat"] = feat_t
        else:
            g = dgl.graph(([], []), num_nodes=self.next_entity_id)
            g.edata[dgl.ETYPE] = torch.zeros(0, dtype=torch.int64)
            g.edata["feat"] = torch.zeros(0, 1, dtype=torch.float32)

        # Add node features: ensure ALL nodes have features (25-dim)
        total_nodes = g.num_nodes()
        all_feats = torch.zeros(total_nodes, 25, dtype=torch.float32)
        for eid, feat in entity_feats.items():
            if isinstance(eid, (int, np.integer)) and eid < total_nodes:
                all_feats[eid] = feat
        g.ndata["feat"] = all_feats

        # Build independent Chain graph (Flight nodes only + preceded_by edges)
        # Chain GAT uses unified 25-dim features (shared GAT)
        if chain_src:
            g_chain = dgl.graph(
                (torch.tensor(chain_src, dtype=torch.int64),
                 torch.tensor(chain_dst, dtype=torch.int64)),
                num_nodes=n_flights)
            g_chain.edata[dgl.ETYPE] = torch.zeros(len(chain_src), dtype=torch.int64)
            g_chain.edata["feat"] = torch.tensor(chain_feat_val, dtype=torch.float32)
            g_chain.ndata["feat"] = all_feats[:n_flights]
        else:
            g_chain = None

        # Build Network heterogeneous graph (Airport + Flight joint modeling)
        # This graph is fed into the unified Network GAT to learn Airport congestion representations + network-wide delay spillover
        g_network, network_node_feat, airport_flight_map = self._build_network_hetero_graph(
            tabular_df, len(tabular_df))
        g_network.ndata['feat'] = network_node_feat

        return g, time_enc, len(tabular_df), g_chain, g_network, airport_flight_map

    def fit_normalizer(self, tabular_dfs):
        """Compute and store global normalization statistics (from all training samples).

        Args:
            tabular_dfs: list of DataFrames (each DataFrame is one day of data)
        """
        n_cont_feats = len(CONFIG.data.cont_cols)
        feat_sum = np.zeros(n_cont_feats, dtype=np.float64)
        feat_sq_sum = np.zeros(n_cont_feats, dtype=np.float64)
        feat_count = 0
        
        for df in tabular_dfs:
            # Add dynamic columns to each DataFrame (consistent with build() method)
            if "O_DAY_FLIGHTS" not in df.columns:
                airport_flight_count = {}
                if "ORIGIN_INDEX" in df.columns:
                    for _, row in df.iterrows():
                        origin_o = str(row.get("ORIGIN_INDEX", ""))
                        dest_o = str(row.get("DEST_INDEX", ""))
                        airport_flight_count[origin_o] = airport_flight_count.get(origin_o, 0) + 1
                        airport_flight_count[dest_o] = airport_flight_count.get(dest_o, 0) + 1
                df = df.copy()
                df["O_DAY_FLIGHTS"] = df["ORIGIN_INDEX"].apply(
                    lambda x: airport_flight_count.get(str(x), 0))
                df["D_DAY_FLIGHTS"] = df["DEST_INDEX"].apply(
                    lambda x: airport_flight_count.get(str(x), 0))
                df["IS_PEAK_HOUR"] = df["CRS_DEP_TIME_MIN"].apply(
                    lambda t: 1.0 if 7*60 <= float(t) <= 9*60 or 17*60 <= float(t) <= 19*60 else 0.0)
            
            for _, row in df.iterrows():
                cont_feats = []
                for col in CONFIG.data.cont_cols:
                    val = row.get(col, 0.0)
                    cont_feats.append(float(val) if not pd.isna(val) else 0.0)
                
                feat = np.array(cont_feats, dtype=np.float32)
                feat_sum += feat.astype(np.float64)
                feat_sq_sum += (feat.astype(np.float64) ** 2)
                feat_count += 1
        
        self.feat_sum = feat_sum
        self.feat_sq_sum = feat_sq_sum
        self.feat_count = feat_count
        
        print(f"Fitted normalizer with {feat_count} samples, {n_cont_feats} features")
        mean = self.feat_sum / self.feat_count
        var = (self.feat_sq_sum / self.feat_count) - (mean ** 2)
        var = np.maximum(var, 1e-8)
        std = np.sqrt(var)
        print(f"Global mean range: [{mean.min():.4f}, {mean.max():.4f}]")
        print(f"Global std range: [{std.min():.4f}, {std.max():.4f}]")

    def _compute_airport_features(self, df):
        """Compute 5-dim airport features from DataFrame.

        Returns:
            ap_to_idx: dict {airport_code: index}
            n_airports: int
            ap_feat: (n_airports, 5) np.array
        """
        all_airports = set()
        for _, row in df.iterrows():
            all_airports.add(str(row.get('ORIGIN_INDEX', '')))
            all_airports.add(str(row.get('DEST_INDEX', '')))
        all_airports = sorted(a for a in all_airports if a)
        ap_to_idx = {ap: i for i, ap in enumerate(all_airports)}
        n_airports = len(all_airports)

        if n_airports == 0:
            return ap_to_idx, 0, np.zeros((0, 5), dtype=np.float32)

        dep_cnt = np.zeros(n_airports, dtype=np.float32)
        arr_cnt = np.zeros(n_airports, dtype=np.float32)
        peak_dep_cnt = np.zeros(n_airports, dtype=np.float32)
        dest_set = [set() for _ in range(n_airports)]
        crs_dep_sum = np.zeros(n_airports, dtype=np.float32)

        for _, row in df.iterrows():
            o = str(row.get('ORIGIN_INDEX', ''))
            d = str(row.get('DEST_INDEX', ''))
            if o not in ap_to_idx or d not in ap_to_idx:
                continue
            oi, di = ap_to_idx[o], ap_to_idx[d]
            dep_cnt[oi] += 1
            arr_cnt[di] += 1
            crs_dep_min = float(row.get('CRS_DEP_TIME_MIN', 720))
            crs_dep_sum[oi] += crs_dep_min / 1440.0
            if (7 * 60 <= crs_dep_min <= 9 * 60) or (17 * 60 <= crs_dep_min <= 19 * 60):
                peak_dep_cnt[oi] += 1
            dest_set[oi].add(d)

        dep_cnt_max = max(dep_cnt.max(), 1.0)
        arr_cnt_max = max(arr_cnt.max(), 1.0)
        dest_div = np.array([len(s) for s in dest_set], dtype=np.float32)
        dest_div_max = max(dest_div.max(), 1.0)

        peak_ratio = np.zeros_like(dep_cnt)
        avg_crs_dep = np.full_like(dep_cnt, 0.5)
        np.divide(peak_dep_cnt, dep_cnt, where=dep_cnt > 0, out=peak_ratio)
        np.divide(crs_dep_sum, dep_cnt, where=dep_cnt > 0, out=avg_crs_dep)

        ap_feat = np.stack([
            dep_cnt / dep_cnt_max,
            arr_cnt / arr_cnt_max,
            peak_ratio,
            dest_div / dest_div_max,
            avg_crs_dep,
        ], axis=1).astype(np.float32)  # (n_airports, 5)

        return ap_to_idx, n_airports, ap_feat

    def _build_network_hetero_graph(self, tabular_df, n_flights):
        """Build Network heterogeneous graph (Airport + Flight joint modeling)."""
        df = tabular_df.reset_index(drop=True)

        # Collect airports + compute features
        ap_to_idx, n_airports, ap_feat = self._compute_airport_features(df)

        if n_airports == 0:
            g_empty = dgl.graph(([], []), num_nodes=n_flights)
            g_empty.edata[dgl.ETYPE] = torch.zeros(0, dtype=torch.int64)
            g_empty.ndata['feat'] = torch.zeros(n_flights, 25, dtype=torch.float32)
            return g_empty, torch.zeros(n_flights, 25, dtype=torch.float32), {
                'flight_node_offset': 0,
                'origin_ap_ids': torch.zeros(n_flights, dtype=torch.int64),
                'dest_ap_ids': torch.zeros(n_flights, dtype=torch.int64),
            }

        flight_offset = n_airports  # Flight nodes start from n_airports

        # --- Build edges ---
        # Edge type 0: Airport → Airport (route busy level)
        route_cnt = {}
        origin_ap_ids = []
        dest_ap_ids = []
        for fidx, (_, row) in enumerate(df.iterrows()):
            o = str(row.get('ORIGIN_INDEX', ''))
            d = str(row.get('DEST_INDEX', ''))
            if o in ap_to_idx and d in ap_to_idx:
                oi, di = ap_to_idx[o], ap_to_idx[d]
                if oi != di:
                    route_cnt[(oi, di)] = route_cnt.get((oi, di), 0) + 1
            origin_ap_ids.append(ap_to_idx.get(o, 0))
            dest_ap_ids.append(ap_to_idx.get(d, 0))

        # Build heterogeneous graph edge lists
        edges_by_type = {
            0: {'src': [], 'dst': [], 'feat': []},  # Airport→Airport
            1: {'src': [], 'dst': [], 'feat': []},  # Flight→Airport (departure)
            2: {'src': [], 'dst': [], 'feat': []},  # Airport→Flight (arrival)
        }

        # Type 0: Airport → Airport
        if route_cnt:
            max_cnt = max(route_cnt.values())
            for (oi, di), cnt in route_cnt.items():
                edges_by_type[0]['src'].append(oi)
                edges_by_type[0]['dst'].append(di)
                edges_by_type[0]['feat'].append([cnt / max_cnt])

        # Type 1 & 2: Flight ↔ Airport
        for fidx, (oi, di) in enumerate(zip(origin_ap_ids, dest_ap_ids)):
            # Flight fidx → Origin Airport oi (departure)
            edges_by_type[1]['src'].append(flight_offset + fidx)
            edges_by_type[1]['dst'].append(oi)
            edges_by_type[1]['feat'].append([1.0])  # Placeholder, no actual edge features

            # Dest Airport di → Flight fidx (arrival)
            edges_by_type[2]['src'].append(di)
            edges_by_type[2]['dst'].append(flight_offset + fidx)
            edges_by_type[2]['feat'].append([1.0])  # Placeholder, no actual edge features

        # Build homogeneous graph (with etype)
        all_src = []
        all_dst = []
        all_etype = []
        all_feat = []

        for etype, etype_data in edges_by_type.items():
            if len(etype_data['src']) > 0:
                all_src.extend(etype_data['src'])
                all_dst.extend(etype_data['dst'])
                all_etype.extend([etype] * len(etype_data['src']))
                all_feat.extend(etype_data['feat'])

        total_nodes = n_airports + n_flights
        if all_src:
            g_network = dgl.graph(
                (torch.tensor(all_src, dtype=torch.int64),
                 torch.tensor(all_dst, dtype=torch.int64)),
                num_nodes=total_nodes)
            g_network.edata[dgl.ETYPE] = torch.tensor(all_etype, dtype=torch.int64)
            g_network.edata['feat'] = torch.tensor(all_feat, dtype=torch.float32)
        else:
            g_network = dgl.graph(([], []), num_nodes=total_nodes)
            g_network.edata[dgl.ETYPE] = torch.zeros(0, dtype=torch.int64)
            g_network.edata['feat'] = torch.zeros(0, 1, dtype=torch.float32)

        # --- Node features ---
        # Unified 25-dim features (20 flight + 5 airport), Network GAT also uses full features

        # Flight nodes: 20-dim (full features)
        flight_feat = np.zeros((n_flights, 20), dtype=np.float32)
        for fidx, (_, row) in enumerate(df.iterrows()):
            flight_feat[fidx] = self._build_flight_node_feat(row).numpy()

        network_node_feat = np.zeros((n_airports + n_flights, 25), dtype=np.float32)
        # Airport: [zeros_20][ap_5d] (last 5 dims)
        network_node_feat[:n_airports, 20:] = ap_feat
        # Flight:  [cont_20][zeros_5] (first 20 dims)
        network_node_feat[n_airports:, :20] = flight_feat

        airport_flight_map = {
            'flight_node_offset': flight_offset,
            'origin_ap_ids': torch.tensor(origin_ap_ids, dtype=torch.int64),
            'dest_ap_ids': torch.tensor(dest_ap_ids, dtype=torch.int64),
        }

        return g_network, torch.tensor(network_node_feat, dtype=torch.float32), airport_flight_map

    def _build_flight_node_feat(self, row) -> torch.Tensor:
        """Build 20-dim node feature vector for a flight (cont_cols only, no padding)."""
        cont_feats = []
        for col in CONFIG.data.cont_cols:
            val = row.get(col, 0.0)
            cont_feats.append(float(val) if not pd.isna(val) else 0.0)
        
        feat = np.array(cont_feats, dtype=np.float32)
        
        # Normalize using global fixed statistics (from training set)
        if self.feat_sum is not None and self.feat_count > 0:
            mean = self.feat_sum / self.feat_count
            var = (self.feat_sq_sum / self.feat_count) - (mean ** 2)
            var = np.maximum(var, 1e-8)
            std = np.sqrt(var)
            feat = (feat - mean) / std
        
        return torch.tensor(feat, dtype=torch.float32)

    def _day_of_year(self, year: int, month: int, day: int) -> int:
        """Compute 0-indexed day of year."""
        from datetime import datetime
        return datetime(year, month, day).timetuple().tm_yday - 1
