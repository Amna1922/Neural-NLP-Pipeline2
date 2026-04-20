"""
Part 2: Sequence Labeling – POS Tagging & NER (UPDATED)
- Generates POS and NER models with proper evaluation
- Saves all required output files
"""

import os, re, json, random, copy
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
POS_TAGS = ['NOUN','VERB','ADJ','ADV','PRON','DET','CONJ','POST','NUM','PUNC','UNK']
POS2IDX = {t: i for i, t in enumerate(POS_TAGS)}
IDX2POS = {i: t for t, i in POS2IDX.items()}

NER_TAGS = ['O','B-PER','I-PER','B-LOC','I-LOC','B-ORG','I-ORG','B-MISC','I-MISC']
NER2IDX = {t: i for i, t in enumerate(NER_TAGS)}
IDX2NER = {i: t for t, i in NER2IDX.items()}

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Gazetteers
PERSONS = set(['عمران خان','شہباز شریف','نواز شریف','آصف زرداری','بلاول بھٹو',
    'مریم نواز','محمد علی جناح','قائداعظم','علامہ اقبال','ذوالفقار بھٹو','بے نظیر بھٹو',
    'پرویز مشرف','آیوب خان','اسحاق ڈار','محمد کاظم','تنویر ملک'])

LOCATIONS = set(['پاکستان','کراچی','لاہور','اسلام آباد','راولپنڈی','پشاور',
    'کوئٹہ','ملتان','فیصل آباد','گوادر','نوشکی','دالبندین','سندھ','پنجاب','بلوچستان',
    'ایران','افغانستان','امریکہ','چین','کابل','تہران','دہلی'])

ORGANISATIONS = set(['حکومت پاکستان','پاکستان مسلم لیگ','پاکستان تحریک انصاف',
    'پاکستان پیپلز پارٹی','نیپرا','پی ٹی آئی','آئی ایم ایف','اقوام متحدہ','سپریم کورٹ'])

# POS Lexicon
PRON_LIST = {'میں','ہم','آپ','وہ','یہ','تم','جو','کیا','کون','کوئی'}
DET_LIST = {'ایک','دو','تین','چار','پانچ','یہ','وہ','اس','ان','کچھ'}
CONJ_LIST = {'اور','یا','لیکن','مگر','جبکہ','کیونکہ','اگر','تو','پھر'}
POST_LIST = {'میں','پر','سے','کو','کے','کی','نے','تک','بعد','ساتھ'}
PUNC_CHARS = set('،۔؟!؛:.,()')
NUM_RE = re.compile(r'^[\d۰-۹]+$')
VERB_SUFFIXES = ('نا','تا','تی','تے','ہے','ہیں','تھا','تھی','گا','گی')
ADJ_SUFFIXES = ('ا','ی','ے','وار','دار','مند')
ADV_SUFFIXES = ('سے','طرح','لیے')

def pos_tag_token(token):
    if all(c in PUNC_CHARS for c in token): return 'PUNC'
    if NUM_RE.match(token): return 'NUM'
    if token in PRON_LIST: return 'PRON'
    if token in CONJ_LIST: return 'CONJ'
    if token in POST_LIST: return 'POST'
    if token in DET_LIST: return 'DET'
    if token.endswith(VERB_SUFFIXES): return 'VERB'
    if token.endswith(ADJ_SUFFIXES): return 'ADJ'
    if token.endswith(ADV_SUFFIXES): return 'ADV'
    return 'NOUN'

def ner_tag_sentence(tokens):
    tags = ['O'] * len(tokens)
    def match_entity(entity_set, bio_prefix):
        for ent in entity_set:
            ent_tokens = ent.split()
            n = len(ent_tokens)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i+n] == ent_tokens:
                    tags[i] = f'B-{bio_prefix}'
                    for j in range(1, n):
                        tags[i+j] = f'I-{bio_prefix}'
    match_entity(PERSONS, 'PER')
    match_entity(LOCATIONS, 'LOC')
    match_entity(ORGANISATIONS, 'ORG')
    return tags

def load_documents(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    parts = re.split(r'\[\d+\]', content)
    return [p.strip() for p in parts if p.strip()]

def sent_tokenize(text):
    return [s.strip() for s in re.split(r'[۔؟!\.]\s*', text) if len(s.strip()) > 3]

def tokenize(text):
    return re.sub(r'\s+', ' ', text).split()

def prepare_dataset(cleaned_path='cleaned.txt', n_sents=500):
    docs = load_documents(cleaned_path)
    cat_kws = {'Politics': ['وزیر','حکومت','پارلیمان'], 'Sports': ['کرکٹ','میچ','ٹیم'], 
               'Economy': ['بجلی','سولر','بینک'], 'International': ['ایران','امریکہ','معاہدہ'],
               'Health': ['کینسر','وائرس','ہسپتال']}
    
    cat_sents = defaultdict(list)
    for doc in docs:
        cat = 'Other'
        for c, kws in cat_kws.items():
            if any(kw in doc for kw in kws): cat = c; break
        for sent in sent_tokenize(doc):
            toks = tokenize(sent)
            if len(toks) >= 3: cat_sents[cat].append((toks, cat))
    
    selected = []
    for cat in list(cat_sents.keys())[:3]:
        pool = cat_sents[cat]
        selected.extend(random.sample(pool, min(100, len(pool))))
    
    remaining = n_sents - len(selected)
    all_rest = [item for cat, pool in cat_sents.items() if cat not in list(cat_sents.keys())[:3] for item in pool]
    if remaining > 0 and all_rest:
        selected.extend(random.sample(all_rest, min(remaining, len(all_rest))))
    
    random.shuffle(selected)
    selected = selected[:n_sents]
    
    annotated = []
    for toks, cat in selected:
        annotated.append({'tokens': toks, 'pos': [pos_tag_token(t) for t in toks], 
                         'ner': ner_tag_sentence(toks), 'cat': cat})
    
    by_cat = defaultdict(list)
    for item in annotated: by_cat[item['cat']].append(item)
    
    train, val, test = [], [], []
    for cat, items in by_cat.items():
        random.shuffle(items)
        n = len(items)
        train.extend(items[:int(0.7*n)])
        val.extend(items[int(0.7*n):int(0.85*n)])
        test.extend(items[int(0.85*n):])
    
    print(f"Dataset: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test

def build_vocab(data):
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for w, _ in Counter(t for item in data for t in item['tokens']).most_common():
        word2idx[w] = len(word2idx)
    return word2idx

def load_embeddings(word2idx):
    if not os.path.exists('embeddings_w2v.npy'):
        return np.random.randn(len(word2idx), 100).astype(np.float32) * 0.01
    emb = np.load('embeddings_w2v.npy')
    new_emb = np.random.randn(len(word2idx), 100).astype(np.float32) * 0.01
    if os.path.exists('w2v_word2idx.json'):
        with open('w2v_word2idx.json', 'r', encoding='utf-8') as f:
            w2v_dict = json.load(f)
        for w, idx in word2idx.items():
            if w in w2v_dict:
                new_emb[idx] = emb[w2v_dict[w]]
    return new_emb

class SeqDataset(Dataset):
    def __init__(self, data, word2idx, task='pos'):
        tag2idx = POS2IDX if task == 'pos' else NER2IDX
        self.samples = [([word2idx.get(t, 1) for t in item['tokens']], 
                        [tag2idx.get(item[task][i], 0) for i in range(len(item['tokens']))]) 
                       for item in data]
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate(batch):
    ids, tags = zip(*batch)
    lens = [len(x) for x in ids]
    max_len = max(lens)
    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_tags = torch.full((len(batch), max_len), -1, dtype=torch.long)
    for i, (id_seq, tag_seq) in enumerate(zip(ids, tags)):
        padded_ids[i, :len(id_seq)] = torch.tensor(id_seq)
        padded_tags[i, :len(tag_seq)] = torch.tensor(tag_seq)
    return padded_ids, padded_tags, torch.tensor(lens)

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, num_tags, pretrained_emb=None, freeze=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 100, padding_idx=0)
        if pretrained_emb is not None:
            self.embed.weight.data.copy_(torch.tensor(pretrained_emb))
            self.embed.weight.requires_grad = not freeze
        self.lstm = nn.LSTM(100, 128, 2, batch_first=True, bidirectional=True, dropout=0.5)
        self.drop = nn.Dropout(0.5)
        self.linear = nn.Linear(256, num_tags)
    
    def forward(self, ids, tags, lens):
        emb = self.drop(self.embed(ids))
        packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.drop(out)
        logits = self.linear(out)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), tags.view(-1), ignore_index=-1)
        return loss
    
    def predict(self, ids, lens):
        emb = self.drop(self.embed(ids))
        packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.drop(out)
        logits = self.linear(out)
        preds = logits.argmax(dim=-1)
        return [preds[b, :l].tolist() for b, l in enumerate(lens)]

def train_model(model, train_loader, val_loader, device, epochs=30, patience=5):
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_f1, best_state, no_improve = -1, None, 0
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for ids, tags, lens in train_loader:
            ids, tags, lens = ids.to(device), tags.to(device), lens
            opt.zero_grad()
            loss = model(ids, tags, lens)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
        
        model.eval()
        all_t, all_p = [], []
        with torch.no_grad():
            for ids, tags, lens in val_loader:
                ids, tags, lens = ids.to(device), tags.to(device), lens
                preds = model.predict(ids, lens)
                for b, l in enumerate(lens):
                    all_t.extend(tags[b, :l].tolist())
                    all_p.extend(preds[b])
        
        tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
        for t, p in zip(all_t, all_p):
            if t < 0: continue
            if t == p: tp[t] += 1
            else: fp[p] += 1; fn[t] += 1
        f1s = [2*tp[c]/(2*tp[c]+fp[c]+fn[c]+1e-10) for c in set(tp.keys()) | set(fn.keys())]
        f1 = np.mean(f1s) if f1s else 0
        print(f"  Epoch {epoch:3d}  loss={total_loss/len(train_loader):.4f}  val_F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1, best_state, no_improve = f1, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"  Early stopping")
            break
    
    if best_state: model.load_state_dict(best_state)
    return model, best_f1

def save_conll(data, filename, task='pos'):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            for token, tag in zip(item['tokens'], item[task]):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Prepare data
    train, val, test = prepare_dataset('cleaned.txt')
    word2idx = build_vocab(train + val + test)
    print(f"Vocab size: {len(word2idx)}")
    
    # Save word2idx
    with open('w2v_word2idx.json', 'w', encoding='utf-8') as f:
        json.dump(word2idx, f, ensure_ascii=False)
    
    pretrained = load_embeddings(word2idx)
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save CONLL files
    save_conll(train, 'data/pos_train.conll', 'pos')
    save_conll(test, 'data/pos_test.conll', 'pos')
    save_conll(train, 'data/ner_train.conll', 'ner')
    save_conll(test, 'data/ner_test.conll', 'ner')
    print("Saved CONLL files to data/")
    
    # POS Tagging
    print("\n=== POS Tagging ===")
    for freeze, name in [(True, 'POS_frozen'), (False, 'POS_finetuned')]:
        train_ds = SeqDataset(train, word2idx, 'pos')
        val_ds = SeqDataset(val, word2idx, 'pos')
        test_ds = SeqDataset(test, word2idx, 'pos')
        
        train_loader = DataLoader(train_ds, 32, True, collate_fn=collate)
        val_loader = DataLoader(val_ds, 32, False, collate_fn=collate)
        test_loader = DataLoader(test_ds, 32, False, collate_fn=collate)
        
        model = BiLSTM(len(word2idx), len(POS_TAGS), pretrained, freeze).to(device)
        model, _ = train_model(model, train_loader, val_loader, device)
        
        # Test
        model.eval()
        all_t, all_p = [], []
        with torch.no_grad():
            for ids, tags, lens in test_loader:
                ids, tags, lens = ids.to(device), tags.to(device), lens
                preds = model.predict(ids, lens)
                for b, l in enumerate(lens):
                    all_t.extend(tags[b, :l].tolist())
                    all_p.extend(preds[b])
        
        correct = sum(1 for t, p in zip(all_t, all_p) if t == p and t >= 0)
        total = sum(1 for t in all_t if t >= 0)
        acc = correct / total if total > 0 else 0
        
        tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
        for t, p in zip(all_t, all_p):
            if t < 0: continue
            if t == p: tp[t] += 1
            else: fp[p] += 1; fn[t] += 1
        f1s = [2*tp[c]/(2*tp[c]+fp[c]+fn[c]+1e-10) for c in set(tp.keys()) | set(fn.keys())]
        f1 = np.mean(f1s) if f1s else 0
        print(f"\n{name} → Accuracy={acc:.4f}  Macro-F1={f1:.4f}")
        
        # Confusion matrix
        cm = np.zeros((len(POS_TAGS), len(POS_TAGS)), dtype=int)
        for t, p in zip(all_t, all_p):
            if 0 <= t < len(POS_TAGS) and 0 <= p < len(POS_TAGS):
                cm[t, p] += 1
        print("Confusion matrix:")
        print("Tags:", POS_TAGS)
        print(cm)
        
        # Save model
        torch.save(model.state_dict(), f'models/bilstm_{name.lower()}.pt')
    
    # NER (with class weights)
    print("\n=== NER ===")
    train_ds = SeqDataset(train, word2idx, 'ner')
    val_ds = SeqDataset(val, word2idx, 'ner')
    test_ds = SeqDataset(test, word2idx, 'ner')
    
    train_loader = DataLoader(train_ds, 16, True, collate_fn=collate)
    val_loader = DataLoader(val_ds, 16, False, collate_fn=collate)
    test_loader = DataLoader(test_ds, 16, False, collate_fn=collate)
    
    class NERModel(nn.Module):
        def __init__(self, vocab_size, pretrained_emb):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, 100, padding_idx=0)
            if pretrained_emb is not None:
                self.embed.weight.data.copy_(torch.tensor(pretrained_emb))
            self.lstm = nn.LSTM(100, 128, 2, batch_first=True, bidirectional=True, dropout=0.5)
            self.drop = nn.Dropout(0.5)
            self.linear = nn.Linear(256, len(NER_TAGS))
        
        def forward(self, ids, tags, lens):
            emb = self.drop(self.embed(ids))
            packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            out = self.drop(out)
            logits = self.linear(out)
            # Class weights for NER
            weights = torch.ones(len(NER_TAGS)).to(ids.device)
            for i in range(1, len(NER_TAGS)):
                weights[i] = 20.0
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), tags.view(-1), 
                                               ignore_index=-1, weight=weights)
            return loss
        
        def predict(self, ids, lens):
            emb = self.drop(self.embed(ids))
            packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            out = self.drop(out)
            logits = self.linear(out)
            preds = logits.argmax(dim=-1)
            return [preds[b, :l].tolist() for b, l in enumerate(lens)]
    
    model = NERModel(len(word2idx), pretrained).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_f1, best_state, no_improve = -1, None, 0
    
    for epoch in range(1, 51):
        model.train()
        total_loss = 0
        for ids, tags, lens in train_loader:
            ids, tags, lens = ids.to(device), tags.to(device), lens
            opt.zero_grad()
            loss = model(ids, tags, lens)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
        
        model.eval()
        all_t, all_p = [], []
        with torch.no_grad():
            for ids, tags, lens in val_loader:
                ids, tags, lens = ids.to(device), tags.to(device), lens
                preds = model.predict(ids, lens)
                for b, l in enumerate(lens):
                    all_t.extend(tags[b, :l].tolist())
                    all_p.extend(preds[b])
        
        tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
        for t, p in zip(all_t, all_p):
            if t < 0: continue
            if t == p: tp[t] += 1
            else: fp[p] += 1; fn[t] += 1
        f1s = [2*tp[c]/(2*tp[c]+fp[c]+fn[c]+1e-10) for c in set(tp.keys()) | set(fn.keys())]
        f1 = np.mean(f1s) if f1s else 0
        print(f"  Epoch {epoch:3d}  loss={total_loss/len(train_loader):.4f}  val_F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1, best_state, no_improve = f1, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if no_improve >= 10:
            print(f"  Early stopping")
            break
    
    if best_state: model.load_state_dict(best_state)
    
    # Test NER
    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for ids, tags, lens in test_loader:
            ids, tags, lens = ids.to(device), tags.to(device), lens
            preds = model.predict(ids, lens)
            for b, l in enumerate(lens):
                all_t.extend(tags[b, :l].tolist())
                all_p.extend(preds[b])
    
    def extract_spans(seq):
        spans = []
        cur, start = None, None
        for i, t in enumerate(seq):
            tag = IDX2NER.get(t, 'O')
            if tag.startswith('B-'):
                if cur: spans.append((cur, start, i-1))
                cur, start = tag[2:], i
            elif tag.startswith('I-') and cur == tag[2:]:
                pass
            else:
                if cur: spans.append((cur, start, i-1))
                cur, start = None, None
        if cur: spans.append((cur, start, len(seq)-1))
        return set(spans)
    
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    gold_spans = extract_spans(all_t)
    pred_spans = extract_spans(all_p)
    
    for span in gold_spans:
        if span in pred_spans: tp[span[0]] += 1
        else: fn[span[0]] += 1
    for span in pred_spans:
        if span not in gold_spans: fp[span[0]] += 1
    
    print("\n=== NER Entity-level F1 ===")
    for t in sorted(set(tp.keys()) | set(fp.keys()) | set(fn.keys())):
        p = tp[t] / (tp[t] + fp[t] + 1e-10)
        r = tp[t] / (tp[t] + fn[t] + 1e-10)
        f = 2*p*r / (p + r + 1e-10)
        print(f"  {t:6s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")
    
    # Save NER model
    torch.save(model.state_dict(), 'models/bilstm_ner.pt')
    print("\n✓ Models saved to models/")
    print("✓ CONLL files saved to data/")
    print("✓ Part 2 complete.")

if __name__ == '__main__':
    main()