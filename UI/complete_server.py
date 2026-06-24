"""
Complete Flight Delay Prediction GUI Server
- Starts with CPU mode (no model loading)
- Loads GPU model in background
- Switches to GPU when ready
- Shows clear status

Run with: python complete_server.py
"""
from flask import Flask, jsonify, send_file, request
import os
import sys
import pickle
import torch
import threading
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'src'))

app = Flask(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DELAY_MEAN = 5.9
DELAY_STD = 23.6

STAGE1_MODEL = None
KG_BUILDER = None
GPU_READY = False
GPU_LOADING = False

def initialize_gpu_async():
    """Load GPU model in background thread"""
    global STAGE1_MODEL, KG_BUILDER, GPU_READY, GPU_LOADING
    
    if GPU_LOADING or GPU_READY:
        print("GPU model already loaded or loading...")
        return
    
    GPU_LOADING = True
    print("\n" + "=" * 60)
    print("Loading GPU model in background...")
    print("This will take about 30-60 seconds...")
    print("=" * 60)
    
    try:
        from src.data.aeolus_dataset import TabularDataset
        from src.data.kg_builder import DailyKGBuilder
        from src.models.stage1 import Stage1Model
        
        stage1_ckpt = os.path.join(_project_root, 'Output', 'src', 'stage1_best.pt')
        STAGE1_MODEL = Stage1Model()
        state = torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
        STAGE1_MODEL.load_state_dict(state.get('model_state', state))
        STAGE1_MODEL.eval().to(DEVICE)
        
        KG_BUILDER = DailyKGBuilder()
        normalizer_path = os.path.join(_project_root, 'Output', 'src', 'normalizer.pkl')
        if os.path.exists(normalizer_path):
            with open(normalizer_path, 'rb') as f:
                NORMALIZER_STATS = pickle.load(f)
            KG_BUILDER.feat_sum = NORMALIZER_STATS['feat_sum']
            KG_BUILDER.feat_sq_sum = NORMALIZER_STATS['feat_sq_sum']
            KG_BUILDER.feat_count = NORMALIZER_STATS['feat_count']
        
        GPU_READY = True
        GPU_LOADING = False
        print("\n" + "=" * 60)
        print("✅ GPU model loaded successfully!")
        print("Now using REAL AI predictions!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        GPU_LOADING = False
        print(f"\n❌ Failed to load GPU model: {e}")
        import traceback
        traceback.print_exc()

def format_time(minutes):
    if minutes == 'Unknown' or minutes is None:
        return 'Unknown'
    try:
        minutes = int(float(minutes))
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    except:
        return str(minutes)

@app.route('/')
def index():
    return send_file('compact_gui.html')

@app.route('/simple')
def simple():
    return send_file('simple_test.html')

@app.route('/status')
def status():
    """Check if GPU is ready"""
    return jsonify({
        'gpu_ready': GPU_READY,
        'gpu_loading': GPU_LOADING,
        'mode': 'GPU (real predictions)' if GPU_READY else ('GPU (loading...)' if GPU_LOADING else 'CPU (mock)'),
        'device': str(DEVICE)
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    year = data.get('year')
    month = data.get('month')
    day = data.get('day')
    flight_num = data.get('flight_num')
    
    if not all([year, month, day, flight_num]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    if GPU_READY and STAGE1_MODEL and KG_BUILDER:
        # GPU mode - real predictions
        return predict_gpu(year, month, day, flight_num)
    else:
        # CPU mode - mock predictions
        return predict_mock(year, month, day, flight_num)

def predict_mock(year, month, day, flight_num):
    """Return mock predictions (for quick testing)"""
    return jsonify({
        'flight_number': str(flight_num),
        'origin': 'LAX',
        'destination': 'SFO',
        'scheduled_dep': '08:00',
        'scheduled_arr': '10:30',
        'delay_probability': 0.45,
        'delay_classification': 'ON-TIME',
        'predicted_delay': 12.5,
        'root_causes': [
            'Previous flight departure delay 25 min - may propagate delay',
            'Origin airport congestion: 35 min cumulative delay in past 2 hours'
        ],
        'ground_truth': {
            'delay_minutes': 15.2,
            'is_delayed': True
        },
        '_note': 'This is MOCK data. Real predictions will be available when GPU model is loaded.'
    })

def predict_gpu(year, month, day, flight_num):
    """Real predictions using GPU model"""
    global STAGE1_MODEL, KG_BUILDER
    
    try:
        from src.data.aeolus_dataset import TabularDataset
        
        dataset = TabularDataset(year, [month])
        df = dataset.get_daily_batches(year, month, day)
        
        if df is None or df.empty:
            return jsonify({'error': f'No data available for {year}/{month}/{day}. Please check the date.'}), 404
        
        # Find flight
        flight_num_str = str(flight_num).strip().upper()
        if 'OP_CARRIER_FL_NUM' in df.columns:
            df['FLIGHT_STR'] = df['OP_CARRIER'].astype(str) + df['OP_CARRIER_FL_NUM'].astype(str).str.zfill(4)
            matches = df[df['FLIGHT_STR'].str.upper() == flight_num_str]
        else:
            matches = df[df['OP_CARRIER_FL_NUM'].astype(str).str.upper() == flight_num_str]
        
        if matches.empty:
            # Try partial match
            partial_num = flight_num_str.replace('*', '').replace('?', '').replace(' ', '')
            matches = df[df['OP_CARRIER_FL_NUM'].astype(str).str.contains(partial_num, case=False, na=False)]
        
        if matches.empty:
            # Return helpful error message
            available_flights = df['FLIGHT_STR'].head(5).tolist() if 'FLIGHT_STR' in df.columns else df['OP_CARRIER_FL_NUM'].head(5).tolist()
            return jsonify({
                'error': f'Flight {flight_num} not found on {year}/{month}/{day}',
                'hint': 'Available sample flights: ' + ', '.join(str(f) for f in available_flights[:5]),
                'total_flights': len(df)
            }), 404
        
        idx = matches.index[0]
        flight_info = matches.iloc[0].to_dict()
        
        # Build KG
        g, time_enc, n_flights, g_chain, g_network, airport_flight_map = KG_BUILDER.build(
            year=year, month=month, day=day, tabular_df=df
        )
        
        g = g.to(DEVICE)
        time_enc = time_enc.to(DEVICE)
        feat = g.ndata["feat"].to(DEVICE)
        target_nids = torch.arange(n_flights, device=DEVICE)
        
        network_feat = None
        network_edge_feat = None
        airport_map = None
        if g_network is not None:
            g_network = g_network.to(DEVICE)
            network_feat = g_network.ndata['feat'].to(DEVICE)
            if g_network.num_edges() > 0:
                network_edge_feat = g_network.edata.get('feat')
            airport_map = {
                'flight_node_offset': airport_flight_map.get('flight_node_offset', n_flights),
                'origin_ap_ids': airport_flight_map['origin_ap_ids'].to(DEVICE),
                'dest_ap_ids': airport_flight_map['dest_ap_ids'].to(DEVICE),
            }
        
        # Predict
        with torch.no_grad():
            result = STAGE1_MODEL(
                blocks=None,
                g_main=g,
                feat=feat,
                etypes_list=None,
                time_enc=time_enc,
                target_idx=target_nids,
                edge_feat_list=None,
                g_chain=g_chain.to(DEVICE) if g_chain is not None else None,
                chain_feat=feat[:n_flights],
                g_network=g_network,
                network_feat=network_feat,
                network_edge_feat=network_edge_feat,
                airport_flight_map=airport_map,
                flight_nids=target_nids,
            )
        
        cls_logits = result['cls_logits'][idx:idx+1].squeeze(-1)
        reg_pred = result['reg_pred'][idx:idx+1].squeeze(-1)
        
        delay_prob = torch.sigmoid(cls_logits).item()
        predicted_delay = (reg_pred * DELAY_STD + DELAY_MEAN).item()
        
        # Root causes
        causes = []
        if flight_info.get('PREV_DEP_DELAY', 0) > 15:
            causes.append(f"Previous flight departure delay {flight_info.get('PREV_DEP_DELAY', 0):.0f} min")
        if flight_info.get('ORIGIN_CUM_DELAY_2H', 0) > 30:
            causes.append(f"Origin airport congestion: {flight_info.get('ORIGIN_CUM_DELAY_2H', 0):.0f} min")
        
        return jsonify({
            'flight_number': f"{flight_info.get('OP_CARRIER', '')}{flight_info.get('OP_CARRIER_FL_NUM', '')}",
            'origin': flight_info.get('ORIGIN_INDEX', 'Unknown'),
            'destination': flight_info.get('DEST_INDEX', 'Unknown'),
            'scheduled_dep': format_time(flight_info.get('CRS_DEP_TIME_MIN', 'Unknown')),
            'scheduled_arr': format_time(flight_info.get('CRS_ARR_TIME_MIN', 'Unknown')),
            'delay_probability': delay_prob,
            'delay_classification': "DELAYED" if delay_prob >= 0.5 else "ON-TIME",
            'predicted_delay': max(0, predicted_delay),
            'root_causes': causes if causes else ['No significant delay factors identified'],
            'ground_truth': {
                'delay_minutes': float(flight_info.get('DEP_DELAY', 0)),
                'is_delayed': bool(flight_info.get('DEP_DELAY', 0) >= 15)
            },
            '_note': 'REAL AI prediction using GPU!'
        })
        
    except Exception as e:
        print(f"GPU prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("✈️  Flight Departure Delay Prediction System")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    print("\nFeatures:")
    print("✅ Starts immediately with CPU mode (mock data)")
    print("✅ Loads GPU model in background")
    print("✅ Automatically switches to real predictions when ready")
    print("\n💡 Check status at: http://127.0.0.1:5002/status")
    print("=" * 80 + "\n")
    
    # Start GPU loading in background
    gpu_thread = threading.Thread(target=initialize_gpu_async, daemon=True)
    gpu_thread.start()
    
    # Run server on port 5002
    print("Server starting on http://127.0.0.1:5002")
    app.run(debug=False, host='127.0.0.1', port=5002)
