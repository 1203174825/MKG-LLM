import dgl
import torch
from dgl.dataloading import (
    MultiLayerNeighborSampler,
    DataLoader as DGLDataLoader,
)
from src.utils.config import CONFIG


def build_sampler():

    k = CONFIG.kg.neighbor_samples  # 10
    num_layers = CONFIG.kg.gat_layers  # 2

    # Per-layer fan-out: [10, 10] for 2 layers
    fanouts = [k] * num_layers
    sampler = MultiLayerNeighborSampler(fanouts)
    return sampler


def create_dataloader(g: dgl.DGLGraph, target_nids: torch.Tensor,
                      batch_size: int, shuffle: bool = True):

    sampler = build_sampler()

    # Ensure target_nids is on CPU for DGLDataLoader
    target_nids_cpu = target_nids.cpu()

    dataloader = DGLDataLoader(
        g.cpu(),
        target_nids_cpu,
        sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,          # 0 for multi-GPU compatibility
        use_uva=False,          # UVA not needed for single-machine
    )
    return dataloader


def extract_flight_node_ids(g: dgl.DGLGraph) -> torch.Tensor:

    # Simplified: use all nodes as target
    # In production: filter by node type attribute
    nids = torch.arange(g.num_nodes())
    return nids


def extract_edge_types(g: dgl.DGLGraph) -> torch.Tensor:

    if hasattr(g, 'canonical_etypes') and len(g.canonical_etypes) > 1:
        # Heterogeneous graph - concatenate all edge type IDs
        all_etypes = []
        for etype in g.canonical_etypes:
            if dgl.ETYPE in g[etype].edata:
                all_etypes.append(g[etype].edata[dgl.ETYPE])
            else:
                # Use relation type index
                rel_idx = g.get_relation_id(etype)
                all_etypes.append(torch.full((g[etype].num_edges(),), rel_idx, dtype=torch.long))
        return torch.cat(all_etypes) if all_etypes else torch.zeros(0, dtype=torch.long)
    
    if dgl.ETYPE in g.edata:
        return g.edata[dgl.ETYPE]
    # Fallback: assign all edges to type 0
    return torch.zeros(g.num_edges(), dtype=torch.long)
