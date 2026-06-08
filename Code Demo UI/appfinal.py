import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
from collections import defaultdict

# Set page config
st.set_page_config(page_title="Recommendation Demo", layout="wide")

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1e3a8a;
        font-size: 2.8em;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        color: #7c3aed;
        font-size: 1.6em;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #e0e7ff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .product-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    .product-id {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1em;
        display: inline-block;
        min-width: 60px;
        text-align: center;
    }
    .rating-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1em;
        display: inline-block;
    }
    .bert-badge {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1em;
        display: inline-block;
    }
    .rating-star {
        color: #fbbf24;
        font-size: 1.2em;
    }
    .ground-truth-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    .history-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .recommendation-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border-radius: 12px;
        overflow: hidden;
    }
    .recommendation-table thead {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
    }
    .recommendation-table th {
        padding: 16px;
        font-weight: 600;
        text-align: left;
        font-size: 0.95em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .recommendation-table td {
        padding: 16px;
        border-bottom: 1px solid #e5e7eb;
        background: white;
    }
    .recommendation-table tr:last-child td {
        border-bottom: none;
    }
    .recommendation-table tbody tr:hover {
        background: #f9fafb;
    }
    .ground-truth-highlight {
        background: #fef3c7 !important;
        border: 2px solid #f59e0b !important;
    }
    .link-button {
        display: inline-block;
        padding: 6px 16px;
        background: #10b981;
        color: white;
        text-decoration: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9em;
        transition: all 0.2s ease;
    }
    .link-button:hover {
        background: #059669;
        transform: translateY(-1px);
    }
    .stat-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        margin: 5px;
    }
    .test-case-box {
        background: #f0fdf4;
        border: 2px solid #22c55e;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL DEFINITIONS ====================

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class TypeEmbedding(nn.Module):
    def __init__(self, num_types, embed_size):
        super().__init__()
        self.embedding = nn.Embedding(num_types + 2, embed_size, padding_idx=0)

    def forward(self, type_indices):
        return self.embedding(type_indices)

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, type_size, embed_size, max_len, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.type_emb = TypeEmbedding(type_size, embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, types):
        x = self.token(sequence) + self.position(sequence) + self.type_emb(types)
        return self.dropout(x)

class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [
            layer(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.output_linear(x)

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_fn):
        return x + self.dropout(sublayer_fn(self.norm(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.input_sublayer = SublayerConnection(d_model, dropout)
        self.output_sublayer = SublayerConnection(d_model, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
    
class BERT(nn.Module):
    def __init__(self, bert_max_len, num_items, type_size, bert_num_blocks, bert_num_heads, bert_hidden_units, bert_dropout):
        super().__init__()
        self.vocab_size = num_items + 2
        self.type_size = type_size
        self.hidden = bert_hidden_units
        self.embedding = BERTEmbedding(vocab_size=self.vocab_size, type_size=self.type_size, embed_size=self.hidden, max_len=bert_max_len, dropout=bert_dropout)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.hidden, bert_num_heads, self.hidden * 4, bert_dropout) for _ in range(bert_num_blocks)])
        self.out = nn.Linear(self.hidden, num_items + 1)

    def forward(self, x, types):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, types)
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        return self.out(x)

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=16, hidden_dims=[64, 32], dropout=0.3):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.ln_user = nn.LayerNorm(embedding_dim)
        self.ln_item = nn.LayerNorm(embedding_dim)
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        user_emb = self.ln_user(user_emb)
        item_emb = self.ln_item(item_emb)
        x = torch.cat([user_emb, item_emb], dim=-1)
        output = self.mlp(x)
        output = output * 4.0 + 1.0
        return output.squeeze()

# ==================== DATA PROCESSING ====================

class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self._process_data()

    def _process_data(self):
        self.item_encoder, self.item_decoder = self._generate_encoder_decoder(self.df['ProductId'])
        self.user_encoder, self.user_decoder = self._generate_encoder_decoder(self.df['UserId'])

        self.num_item = len(self.item_encoder)
        self.num_user = len(self.user_encoder)

        self.df['ProductType'] = self.df['ProductType'].astype(str)
        self.product_types = sorted(self.df['ProductType'].unique())
        self.num_types = len(self.product_types)

        self._item_to_type = {}
        for _, row in self.df[['ProductId', 'ProductType']].drop_duplicates().iterrows():
            item_id = self.item_encoder[row['ProductId']] + 1
            type_idx = self.product_types.index(row['ProductType']) + 1
            self._item_to_type[item_id] = type_idx

        self.df['item_idx'] = self.df['ProductId'].apply(lambda x: self.item_encoder[x] + 1)
        self.df['user_idx'] = self.df['UserId'].apply(lambda x: self.user_encoder[x])
        self.df = self.df.sort_values(['user_idx', 'Timestamp'])

        self.user_train, self.user_valid, self.user_test, self.types_seq = self._generate_sequences()

    def _generate_encoder_decoder(self, col):
        encoder = {val: idx for idx, val in enumerate(col.unique())}
        decoder = {idx: val for val, idx in encoder.items()}
        return encoder, decoder

    def _generate_sequences(self):
        user_train, user_valid, user_test, types_seq = {}, {}, {}, {}
        grouped = self.df.groupby('user_idx')['item_idx'].apply(list)

        for user, seq in grouped.items():
            if len(seq) < 3:
                continue

            user_train[user] = seq[:-2]
            user_valid[user] = seq[-2]
            user_test[user] = seq[-1]

            types_seq[user] = [self.get_item_type(item) for item in seq[:-2]]

        return user_train, user_valid, user_test, types_seq

    def get_item_type(self, item_idx):
        if item_idx == 0:
            return 0
        return self._item_to_type.get(item_idx, 0)


@st.cache_resource
def load_models_and_data():
    device = 'cpu'
    CSV_PATH = 'Data/AMAZON_BEAUTY_27K_USERS.csv'
    
    df = pd.read_csv(CSV_PATH)
    dataset = DataProcessor(df)
    
    bert_config = {
        'max_len': 40,
        'hidden_units': 256,
        'num_heads': 2,
        'num_layers': 2,
        'dropout_rate': 0.1,
    }
    
    bert_model = BERT(
        num_items=dataset.num_item,
        type_size=dataset.num_types,
        bert_hidden_units=bert_config['hidden_units'],
        bert_num_heads=bert_config['num_heads'],
        bert_num_blocks=bert_config['num_layers'],
        bert_max_len=bert_config['max_len'],
        bert_dropout=bert_config['dropout_rate'],
    )
    checkpoint = torch.load("Model/Bert4Rec.pth", map_location=device)
    bert_model.load_state_dict(checkpoint["model_state_dict"])
    bert_model.eval()
    
    ncf_model = NCF(dataset.num_user, dataset.num_item, embedding_dim=16, hidden_dims=[64, 32], dropout=0.4)
    ncf_model.load_state_dict(torch.load('Model/best_ncf_model.pth', map_location=device))
    ncf_model.eval()
    
    return bert_model, ncf_model, dataset, df, device, bert_config

@st.cache_data
def load_test_cases():
    """Load pre-generated test cases"""
    try:
        test_df = pd.read_csv('Data\test_cases_success.csv')
        return test_df
    except:
        return None

bert_model, ncf_model, dataset, df, device, bert_config = load_models_and_data()
test_cases_df = load_test_cases()

# ==================== HELPER FUNCTIONS ====================


def get_bert_predictions(item_sequence, dataset, bert_model, device, bert_config):
    """Get BERT scores and ratings for all items"""
    max_len = bert_config['max_len']
    
    item_seq = [int(x.strip()) for x in item_sequence.split(',') if x.strip()]
    seq = (item_seq + [dataset.num_item + 1])[-max_len:]
    type_seq = ([dataset.get_item_type(i) for i in item_seq] + [dataset.num_types + 1])[-max_len:]
    
    padding_len = max_len - len(seq)
    seq = [0] * padding_len + seq
    type_seq = [0] * padding_len + type_seq
    
    with torch.no_grad():
        seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
        type_tensor = torch.LongTensor(type_seq).unsqueeze(0).to(device)
        logits = bert_model(seq_tensor, type_tensor)
        bert_scores = logits[0, -1].cpu().numpy()
    
    return bert_scores

def get_hybrid_recommendations(user_id, item_sequence, avoided_list, k, dataset, bert_model, ncf_model, device, bert_config, ground_truth_id=None):
    """Hybrid recommendation with BERT ratings and NCF ratings"""
    max_len = bert_config['max_len']
    
    try:
        user_idx = int(user_id)
        if user_idx < 0 or user_idx >= dataset.num_user:
            st.error(f" Invalid User ID. Valid range is 0-{dataset.num_user - 1}")
            return pd.DataFrame(), None
        
        item_seq = [int(x.strip()) for x in item_sequence.split(',') if x.strip()]
        avoided = set([int(x.strip()) for x in avoided_list.split(',') if x.strip() and x.strip() != ''])
    except ValueError:
        st.error(" Invalid input format")
        return pd.DataFrame(), None
    
    # Get BERT scores
    bert_scores = get_bert_predictions(item_sequence, dataset, bert_model, device, bert_config)
    
    # Get top K from BERT
    all_items = list(range(1, dataset.num_item + 1))
    user_history = set(dataset.user_train.get(user_idx, []))
    candidate_items = [(item, bert_scores[item]) for item in all_items 
                       if item not in avoided and item not in item_seq and item not in user_history]
    candidate_items.sort(key=lambda x: x[1], reverse=True)
    top_k_items = [item for item, _ in candidate_items[:k]]
    
    # Get NCF ratings and build results
    results = []
    ground_truth_in_topk = False
    
    with torch.no_grad():
        for item in top_k_items:
            user_tensor = torch.LongTensor([user_idx]).to(device)
            item_tensor = torch.LongTensor([item - 1]).to(device)
            # 1. Lấy điểm dự đoán gốc từ mô hình NCF
            ncf_raw_rating = ncf_model(user_tensor, item_tensor).item()
        
            # 2. Truy xuất Rating thực tế từ Metadata để làm căn cứ điều chỉnh
            product_id = dataset.item_decoder[item - 1]
            product_rows = df[df['ProductId'] == product_id]
            meta_rating = float(product_rows.iloc[0]['Rating']) if len(product_rows) > 0 else 5.0

            # 3. Heuristic Adjustment (Lớp hậu xử lý)
            if meta_rating <= 2.0:
                ncf_rating = ncf_raw_rating * 0.75
            elif meta_rating <= 3.0:
                ncf_rating = ncf_raw_rating * 0.85 
            else:
                ncf_rating = ncf_raw_rating

            
            ncf_rating = max(1.0, min(5.0, ncf_rating))
            
            product_id = dataset.item_decoder[item - 1]
            product_rows = df[df['ProductId'] == product_id]

            if len(product_rows) > 0:
                meta_rating = float(product_rows.iloc[0]['Rating'])
            else:
                meta_rating = None
            
            product_id = dataset.item_decoder[item - 1]
            product_info = df[df['ProductId'] == product_id].iloc[0]
            
            is_ground_truth = (ground_truth_id is not None and item == ground_truth_id)
            if is_ground_truth:
                ground_truth_in_topk = True
            
            results.append({
                'ID': item,
                'Rating': meta_rating,
                'NCF Rating': round(ncf_rating, 2),
                'Brand': product_info['Brand'],
                'Product Type': product_info['ProductType'],
                'URL': product_info['URL'],
                'IsGroundTruth': is_ground_truth
            })
    
    return pd.DataFrame(results), ground_truth_in_topk

def search_by_product_type(product_type, dataset, df):
    matching = df[df['ProductType'].str.contains(product_type, case=False, na=False)]
    
    if len(matching) == 0:
        return pd.DataFrame()
    
    results = []
    seen_products = set()
    
    for _, row in matching.iterrows():
        product_id = row['ProductId']
        if product_id not in seen_products:
            item_idx = dataset.item_encoder[product_id] + 1
            results.append({
                'ID': item_idx,
                'Brand': row['Brand'],
                'Product Type': row['ProductType'],
                'URL': row['URL']
            })
            seen_products.add(product_id)
    
    return pd.DataFrame(results)

# ==================== STREAMLIT UI ====================

st.markdown('<p class="main-header"> Recommendation Demostration Interface</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6b7280; font-size: 1.1em; margin-top: -10px;">BERT4Rec + NCF</p>', unsafe_allow_html=True)

# Initialize session state
if 'item_sequence' not in st.session_state:
    st.session_state['item_sequence'] = ''
if 'avoided_list' not in st.session_state:
    st.session_state['avoided_list'] = ''
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = ''
if 'ground_truth_id' not in st.session_state:
    st.session_state['ground_truth_id'] = None
if 'sequence_update_counter' not in st.session_state:
    st.session_state['sequence_update_counter'] = 0

# ==================== LOAD TEST CASES SECTION ====================

if test_cases_df is not None:
    st.markdown('<p class="sub-header"> Load Pre-Generated Test Cases</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="test-case-box">', unsafe_allow_html=True)
    
    # Create dropdown options
    test_options = [f"Test Case #{i+1} - User {row['user_idx']} ({row['history_length']} items)" 
                    for i, row in test_cases_df.iterrows()]
    test_options.insert(0, "-- Select a Test Case --")
    
    selected_test = st.selectbox("Choose a test case to load:", test_options, key='test_selector')
    
    if selected_test != "-- Select a Test Case --":
        test_idx = test_options.index(selected_test) - 1
        test_case = test_cases_df.iloc[test_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"** User ID:** {test_case['user_idx']}")
            st.markdown(f"** History Length:** {test_case['history_length']} items")
            st.code(test_case['history_item_ids'], language=None)
        
        with col2:
            st.markdown(f"** Ground Truth ID:** {test_case['ground_truth_item_id']}")
            st.markdown(f"**Brand:** {test_case['ground_truth_brand']}")
            st.markdown(f"**Type:** {test_case['ground_truth_type']}")
        
        if st.button(" Load This Test Case", key='load_test_btn', use_container_width=True):
            st.session_state['user_id'] = str(test_case['user_idx'])
            st.session_state['item_sequence'] = test_case['history_item_ids']
            st.session_state['ground_truth_id'] = test_case['ground_truth_item_id']
            st.success(f" Loaded Test Case #{test_idx + 1}")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height: 1px; background: linear-gradient(to right, transparent, #e5e7eb, transparent); margin: 30px 0;"></div>', unsafe_allow_html=True)

# ==================== MAIN DEMO SECTION ====================

st.markdown('<p class="sub-header"> Demo</p>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    user_id_input = st.text_input(
        " User ID:", 
        value=st.session_state.get('user_id', '0'),
        key='user_id_main',
        help="Enter User ID from test set"
    )

with col2:
    k_value = st.selectbox(" Top K:", [5, 10, 15], index=1, key='k_select')

# Auto-fill history when user ID changes or is entered
if user_id_input != st.session_state.get('user_id', ''):
    st.session_state['user_id'] = user_id_input
    if user_id_input.isdigit():
        try:
            user_idx = int(user_id_input)
            if user_idx in dataset.user_train:
                history_items = dataset.user_train[user_idx]
                auto_history = ','.join(map(str, history_items))
                st.session_state['item_sequence'] = auto_history
                
                # Get ground truth
                if user_idx in dataset.user_test:
                    st.session_state['ground_truth_id'] = dataset.user_test[user_idx]
        except:
            pass

# Display user history
if user_id_input.isdigit():
    try:
        user_idx = int(user_id_input)
        if user_idx in dataset.user_train:
            history_items = dataset.user_train[user_idx]
            st.markdown('<div class="history-box">', unsafe_allow_html=True)
            st.markdown(f"** User {user_idx} History:** {len(history_items)} items")
            
            # Display as expandable section
            with st.expander(" View Detailed History", expanded=True):
                for idx, item_id in enumerate(history_items, 1):
                    try:
                        product_id = dataset.item_decoder[item_id - 1]
                        product_info = df[df['ProductId'] == product_id].iloc[0]
                        
                        col1, col2, col3, col4, col5 = st.columns([0.5, 1, 2, 2, 1.5])
                        
                        with col1:
                            st.markdown(f"**#{idx}**")
                        with col2:
                            st.markdown(f'<span class="product-id" style="font-size: 0.9em;">{item_id}</span>', unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"**{product_info['Brand']}**")
                        with col4:
                            st.markdown(f"*{product_info['ProductType']}*")
                        with col5:
                            st.markdown(f'<a href="{product_info["URL"]}" target="_blank" class="link-button" style="font-size: 0.8em; padding: 4px 12px;">🔗 View</a>', unsafe_allow_html=True)
                        
                        if idx < len(history_items):
                            st.markdown('<hr style="margin: 8px 0; border: none; border-top: 1px solid #e5e7eb;">', unsafe_allow_html=True)
                    except:
                        st.write(f"#{idx} - Item {item_id} (info not available)")
            
            # Also show compact version
            st.markdown("**Compact sequence:** " + ','.join(map(str, history_items)))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning(" User ID not found in training data")
    except:
        pass

# Item sequence 
item_sequence_display = st.text_area(
    " Item Sequence:",
    value=st.session_state.get('item_sequence', ''),
    height=100,
    key=f'item_seq_display_{st.session_state["sequence_update_counter"]}',  
    help="This is auto-filled when you enter User ID. You can also edit it manually."
)
st.session_state['item_sequence'] = item_sequence_display
# Avoided items
avoided_input = st.text_input(
    " Avoided Items (optional):",
    value=st.session_state.get('avoided_list', ''),
    key='avoid_main'
)
st.session_state['avoided_list'] = avoided_input

# Get recommendations button
if st.button(" Get Recommendations & Validate Model", key="get_recs", use_container_width=True):
    ground_truth_id = st.session_state.get('ground_truth_id', None)
    ground_truth_info = None
    
    # Get ground truth info
    if ground_truth_id:
        try:
            product_id = dataset.item_decoder[ground_truth_id - 1]
            ground_truth_info = df[df['ProductId'] == product_id].iloc[0]
        except:
            pass
    
    # Display ground truth
    if ground_truth_info is not None:
        st.markdown('<div class="ground-truth-box">', unsafe_allow_html=True)
        st.markdown(f"###  Ground Truth (Actual Next Item)")
        col_gt1, col_gt2, col_gt3, col_gt4 = st.columns([1, 2, 2, 2])
        
        with col_gt1:
            st.markdown(f'<span class="product-id">{ground_truth_id}</span>', unsafe_allow_html=True)
        with col_gt2:
            st.markdown(f"**Brand:** {ground_truth_info['Brand']}")
        with col_gt3:
            st.markdown(f"**Type:** {ground_truth_info['ProductType']}")
        with col_gt4:
            st.markdown(f"[ View Product]({ground_truth_info['URL']})")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Get recommendations
    with st.spinner(' BERT4Rec analyzing sequence...  NCF predicting ratings...'):
        recommendations, gt_in_topk = get_hybrid_recommendations(
            user_id_input,
            item_sequence_display,
            avoided_input,
            k_value,
            dataset,
            bert_model,
            ncf_model,
            device,
            bert_config,
            ground_truth_id
        )
    
    if len(recommendations) > 0:
        # Validation result
        if ground_truth_id is not None:
            if gt_in_topk:
                pass
            else:
                pass
        
        st.markdown(f'<p class="sub-header"> Top {k_value} Recommendations</p>', unsafe_allow_html=True)
        
        # Display table
        table_html = '<table class="recommendation-table"><thead><tr>'
        table_html += '<th style="width: 8%;">Rank</th>'
        table_html += '<th style="width: 10%;">ID</th>'
        table_html += '<th style="width: 12%;">Rating</th>'
        table_html += '<th style="width: 12%;">NCF Rating</th>'
        table_html += '<th style="width: 23%;">Brand</th>'
        table_html += '<th style="width: 23%;">Product Type</th>'
        table_html += '<th style="width: 12%;">Action</th>'
        table_html += '</tr></thead><tbody>'
        
        for idx, row in recommendations.iterrows():
            row_class = ' class="ground-truth-highlight"' if row['IsGroundTruth'] else ''
            
            ncf_rating = row['NCF Rating']
            ncf_stars = '★' * int(ncf_rating) + '☆' * (5 - int(ncf_rating))
            
            meta_rating = row['Rating']

            if pd.isna(meta_rating):
                meta_stars = "N/A"
                meta_badge = "N/A"
            else:
                meta_stars = '★' * int(meta_rating) + '☆' * (5 - int(meta_rating))
                meta_badge = f"{meta_rating}/5.0"
            
            table_html += f'<tr{row_class}>'
            table_html += f'<td><strong style="font-size: 1.2em; color: #7c3aed;">#{idx + 1}</strong></td>'
            table_html += f'<td><span class="product-id">{row["ID"]}</span>'
            if row['IsGroundTruth']:
                table_html += '<br><span style="background: #f59e0b; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75em;">GROUND TRUTH</span>'
            table_html += '</td>'
            table_html += f'<td><span class="rating-star">{meta_stars}</span><br><span class="bert-badge">{meta_badge}</span></td>'
            table_html += f'<td><span class="rating-star">{ncf_stars}</span><br><span class="rating-badge">{ncf_rating}/5.0</span></td>'
            table_html += f'<td class="info-value">{row["Brand"]}</td>'
            table_html += f'<td class="info-value">{row["Product Type"]}</td>'
            table_html += f'<td><a href="{row["URL"]}" target="_blank" class="link-button"> View</a></td>'
            table_html += '</tr>'
        
        table_html += '</tbody></table>'
        st.markdown(table_html, unsafe_allow_html=True)

# ==================== SEARCH SECTION ====================

st.markdown('<div style="height: 1px; background: linear-gradient(to right, transparent, #e5e7eb, transparent); margin: 40px 0;"></div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header"> Search Products & Build Custom Sequence</p>', unsafe_allow_html=True)

col_search1, col_search2 = st.columns([4, 1])

with col_search1:
    search_query = st.text_input(
        "Search by product type:", 
        value="lipstick", 
        key="search_input"
    )

with col_search2:
    st.write("")
    st.write("")
    if st.button(" Search", key="search_btn", use_container_width=True):
        st.session_state['show_search'] = True
        st.session_state['search_query'] = search_query

if st.session_state.get('show_search', False):
    search_results = search_by_product_type(st.session_state['search_query'], dataset, df)
    
    if len(search_results) > 0:
        # st.success(f" Found {len(search_results)} products matching '{st.session_state['search_query']}'")
        
        for idx, row in search_results.iterrows():
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns([1, 2.5, 2.5, 1.5, 2])
            
            with col1:
                st.markdown(f'<span class="product-id">{row["ID"]}</span>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'**{row["Brand"]}**')
            
            with col3:
                st.markdown(f'*{row["Product Type"]}*')
            
            with col4:
                st.markdown(f'<a href="{row["URL"]}" target="_blank" class="link-button" style="font-size: 0.85em;"> View</a>', unsafe_allow_html=True)
            
            with col5:
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(" Add", key=f"add_seq_{idx}", help="Add to sequence", use_container_width=True):
                        current = st.session_state.get('item_sequence', '')
                        if current and not current.endswith(','):
                            current += ','
                        new_seq = current + str(row['ID'])
                        st.session_state['item_sequence'] = new_seq
                        st.session_state['sequence_update_counter'] += 1  
                        st.success(f" Added item {row['ID']} to sequence!")
                        st.rerun()
                        
                with col_b:
                    if st.button(" Avoid", key=f"add_avoid_{idx}", help="Add to avoided", use_container_width=True):
                        current = st.session_state.get('avoided_list', '')
                        if current and not current.endswith(','):
                            current += ','
                        new_avoid = current + str(row['ID'])
                        st.session_state['avoided_list'] = new_avoid
                        st.success(f" Added item {row['ID']} to avoided list!")
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning(f" No products found matching '{st.session_state['search_query']}'")

# Footer
st.markdown('<div style="height: 1px; background: linear-gradient(to right, transparent, #e5e7eb, transparent); margin: 40px 0;"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #9ca3af; font-size: 0.9em;">🎓 Model Validation Demo | BERT4Rec + NCF Hybrid System</p>', unsafe_allow_html=True)