"""
Part 1: Word Embeddings
- TF-IDF Weighted Representations
- PMI Weighted Representations  
- Skip-gram Word2Vec (from scratch in PyTorch)
"""

import os, json, re, math, random, time
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ─────────────────────────────────────────────
# 0.  Utilities
# ─────────────────────────────────────────────
def load_documents(filepath):
    """Parse [N]\n...\n[N+1]\n... format into list of doc strings."""
    docs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # split on document markers [1], [2], ...
    parts = re.split(r'\[\d+\]', content)
    for p in parts:
        p = p.strip()
        if p:
            docs.append(p)
    return docs

def tokenize(text):
    """Simple whitespace + punctuation tokenizer for Urdu/Roman text."""
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    return text.split()

def build_vocab(all_tokens, max_vocab=10000):
    """Return word2idx, idx2word restricted to top max_vocab-1 + <UNK>."""
    counts = Counter(all_tokens)
    most_common = [w for w, _ in counts.most_common(max_vocab - 1)]
    word2idx = {'<UNK>': 0}
    for w in most_common:
        word2idx[w] = len(word2idx)
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, counts


# ─────────────────────────────────────────────
# 1.1  TF-IDF
# ─────────────────────────────────────────────
def compute_tfidf(docs, word2idx, max_vocab=10000):
    V = len(word2idx)
    N = len(docs)
    # tokenize each doc
    tokenized = [tokenize(d) for d in docs]
    # map to indices
    indexed  = [[word2idx.get(t, 0) for t in tok] for tok in tokenized]

    # TF : raw count / doc length
    tf_matrix = np.zeros((N, V), dtype=np.float32)
    for d, ids in enumerate(indexed):
        if len(ids) == 0: continue
        cnt = Counter(ids)
        for idx, c in cnt.items():
            tf_matrix[d, idx] = c / len(ids)

    # DF
    df = np.zeros(V, dtype=np.float32)
    for ids in indexed:
        for idx in set(ids):
            df[idx] += 1

    # IDF
    idf = np.log(N / (1 + df))   # shape (V,)

    tfidf = tf_matrix * idf[np.newaxis, :]   # (N, V)
    return tfidf, idf

def top10_per_category(tfidf_matrix, docs, word2idx, idx2word, meta_path='Metadata.json'):
    """Report top-10 discriminative words per topic category."""
    if not os.path.exists(meta_path):
        print("Metadata.json not found – skipping category analysis.")
        return

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # Build simple keyword-based category labels matching our doc indices
    category_keywords = {
        'Politics':          ['election','government','minister','parliament','وزیر','حکومت','انتخاب'],
        'Sports':            ['cricket','match','team','player','score','کرکٹ','میچ'],
        'Economy':           ['inflation','trade','bank','GDP','budget','بجلی','سولر','اقتصاد'],
        'International':     ['UN','treaty','foreign','bilateral','conflict','ایران','امریکہ'],
        'Health & Society':  ['hospital','disease','vaccine','flood','education','کینسر','وائرس'],
    }

    idx2word_local = idx2word
    doc_titles = [meta.get(str(i+1), {}).get('title','') for i in range(len(docs))]

    cat_doc_indices = defaultdict(list)
    for d, title in enumerate(doc_titles):
        assigned = False
        for cat, kws in category_keywords.items():
            if any(kw in title or kw in docs[d] for kw in kws):
                cat_doc_indices[cat].append(d)
                assigned = True
                break
        if not assigned:
            cat_doc_indices['Other'].append(d)

    print("\n=== Top-10 TF-IDF Words per Category ===")
    for cat, d_ids in cat_doc_indices.items():
        if not d_ids: continue
        avg_tfidf = tfidf_matrix[d_ids].mean(axis=0)
        top_ids = np.argsort(avg_tfidf)[::-1][:10]
        words = [idx2word_local.get(i, '<UNK>') for i in top_ids]
        print(f"  {cat}: {words}")


# ─────────────────────────────────────────────
# 1.2  PPMI
# ─────────────────────────────────────────────
def compute_ppmi(docs, word2idx, window=5):
    V = len(word2idx)
    # build co-occurrence
    cooc = np.zeros((V, V), dtype=np.float32)
    for doc in docs:
        tokens = tokenize(doc)
        ids = [word2idx.get(t, 0) for t in tokens]
        for i, w in enumerate(ids):
            lo = max(0, i - window)
            hi = min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    cooc[w, ids[j]] += 1.0

    total = cooc.sum()
    if total == 0:
        return cooc
    p_w = cooc.sum(axis=1) / total          # P(w)
    p_c = cooc.sum(axis=0) / total          # P(c)
    p_wc = cooc / total                     # P(w,c)

    # PPMI = max(0, log2(P(w,c)/(P(w)*P(c))))
    denom = np.outer(p_w, p_c) + 1e-10
    pmi   = np.log2(p_wc / denom + 1e-10)
    ppmi  = np.maximum(0, pmi)
    return ppmi

def tsne_visualise(ppmi_matrix, idx2word, word2idx, counts, top_n=200, save_path='tsne_ppmi.png'):
    """2-D t-SNE of top_n tokens, colour-coded by rough semantic category."""
    freq_words = sorted(word2idx.keys(), key=lambda w: counts.get(w, 0), reverse=True)
    freq_words = [w for w in freq_words if w != '<UNK>'][:top_n]

    vecs = np.array([ppmi_matrix[word2idx[w]] for w in freq_words])
    # reduce dim first
    if vecs.shape[1] > 50:
        from sklearn.decomposition import TruncatedSVD
        vecs = TruncatedSVD(n_components=50, random_state=42).fit_transform(vecs)

    # tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(vecs)

    # rough colour categories
    cat_map = {
        'politics':   ['وزیر','حکومت','پارلیمان','الیکشن','سیاسی','صدر','وزیراعظم'],
        'sports':     ['کرکٹ','میچ','ٹیم','کھلاڑی','اسکور','بیٹنگ'],
        'geography':  ['پاکستان','کراچی','لاہور','اسلام','بلوچستان','سندھ'],
        'economy':    ['بجلی','سولر','بینک','قیمت','مہنگائی','بجٹ'],
        'health':     ['کینسر','وائرس','ہسپتال','علاج','بیماری'],
    }
    color_list = ['#e41a1c','#377eb8','#4daf4a','#ff7f00','#984ea3']
    cat_names  = list(cat_map.keys())

    word_cats = []
    for w in freq_words:
        assigned = 'other'
        for cat, kws in cat_map.items():
            if any(kw in w for kw in kws):
                assigned = cat
                break
        word_cats.append(assigned)

    fig, ax = plt.subplots(figsize=(14, 10))
    plotted = set()
    for i, (x, y) in enumerate(coords):
        cat = word_cats[i]
        if cat in cat_names:
            c = color_list[cat_names.index(cat)]
        else:
            c = 'grey'
        lbl = cat if cat not in plotted else None
        plotted.add(cat)
        ax.scatter(x, y, c=c, alpha=0.7, s=20, label=lbl)
        ax.annotate(freq_words[i], (x, y), fontsize=6, alpha=0.8)

    ax.legend(loc='best')
    ax.set_title('t-SNE of Top-200 Tokens (PPMI)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved t-SNE plot → {save_path}")

def nearest_neighbours_cosine(matrix, word2idx, idx2word, query_words, top_k=5):
    """Report top_k nearest neighbours by cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normed = matrix / norms
    print("\n=== PPMI Nearest Neighbours ===")
    for qw in query_words:
        if qw not in word2idx:
            print(f"  '{qw}' not in vocab")
            continue
        qi = word2idx[qw]
        sims = normed @ normed[qi]
        top_ids = np.argsort(sims)[::-1][1:top_k+1]
        neighbours = [idx2word[i] for i in top_ids]
        print(f"  {qw}: {neighbours}")


# ─────────────────────────────────────────────
# 2.1  Skip-gram Word2Vec Dataset
# ─────────────────────────────────────────────
class SkipGramDataset(Dataset):
    def __init__(self, all_tokens_ids, window=5):
        self.pairs = []
        for i, center in enumerate(all_tokens_ids):
            lo = max(0, i - window)
            hi = min(len(all_tokens_ids), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    self.pairs.append((center, all_tokens_ids[j]))
        self.pairs = torch.tensor(self.pairs, dtype=torch.long)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx, 0], self.pairs[idx, 1]


class SkipGramModel(nn.Module):
    """Skip-gram with negative sampling (no built-in attention/transformer)."""
    def __init__(self, vocab_size, embed_dim=100):
        super().__init__()
        self.V = nn.Embedding(vocab_size, embed_dim)  # centre embeddings
        self.U = nn.Embedding(vocab_size, embed_dim)  # context embeddings
        nn.init.uniform_(self.V.weight, -0.5/embed_dim, 0.5/embed_dim)
        nn.init.zeros_(self.U.weight)

    def forward(self, center, context, negatives):
        # center:    (B,)
        # context:   (B,)
        # negatives: (B, K)
        vc   = self.V(center)           # (B, d)
        uo   = self.U(context)          # (B, d)
        u_neg = self.U(negatives)       # (B, K, d)

        pos_score = torch.sum(uo * vc, dim=1)          # (B,)
        neg_score = torch.bmm(u_neg, vc.unsqueeze(2))  # (B, K, 1)
        neg_score = neg_score.squeeze(2)               # (B, K)

        loss = -torch.mean(
            torch.log(torch.sigmoid(pos_score) + 1e-10) +
            torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=1)
        )
        return loss


def build_noise_dist(counts, word2idx, alpha=0.75):
    vocab_size = len(word2idx)
    freq = np.zeros(vocab_size)
    for w, c in counts.items():
        if w in word2idx:
            freq[word2idx[w]] = c
    freq = freq ** alpha
    freq /= freq.sum()
    return freq


def train_skipgram(docs, word2idx, idx2word, counts,
                   embed_dim=100, window=5, K=10, epochs=5,
                   batch_size=512, lr=0.001, device='cpu',
                   save_path='embeddings_w2v.npy',
                   loss_plot_path='w2v_loss.png'):

    # all token IDs
    all_tokens = []
    for doc in docs:
        all_tokens.extend([word2idx.get(t, 0) for t in tokenize(doc)])

    dataset = SkipGramDataset(all_tokens, window=window)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    noise_freq = build_noise_dist(counts, word2idx)
    vocab_size = len(word2idx)

    model     = SkipGramModel(vocab_size, embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    all_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for step, (center, context) in enumerate(loader):
            center  = center.to(device)
            context = context.to(device)
            # sample negatives
            neg_ids = np.random.choice(vocab_size,
                                       size=(center.size(0), K),
                                       p=noise_freq)
            negatives = torch.tensor(neg_ids, dtype=torch.long).to(device)

            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if step % 500 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  step {step}  loss={loss.item():.4f}")
                all_losses.append(loss.item())

        print(f"Epoch {epoch+1} avg loss: {epoch_loss/len(loader):.4f}")

    # averaged embeddings
    V_emb = model.V.weight.detach().cpu().numpy()
    U_emb = model.U.weight.detach().cpu().numpy()
    final_emb = (V_emb + U_emb) / 2.0
    np.save(save_path, final_emb)
    print(f"Saved embeddings → {save_path}")

    # loss curve
    plt.figure()
    plt.plot(all_losses)
    plt.xlabel('Step (x500)')
    plt.ylabel('Loss')
    plt.title('Skip-gram Training Loss')
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss curve → {loss_plot_path}")

    return final_emb, model


# ─────────────────────────────────────────────
# 2.2  Evaluation helpers
# ─────────────────────────────────────────────
def top_k_neighbours(embeddings, word2idx, idx2word, query_words, k=10):
    norms  = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms
    print(f"\n=== Top-{k} Nearest Neighbours (W2V) ===")
    for qw in query_words:
        if qw not in word2idx:
            print(f"  '{qw}' not in vocab")
            continue
        qi   = word2idx[qw]
        sims = normed @ normed[qi]
        top_ids = np.argsort(sims)[::-1][1:k+1]
        nbrs = [idx2word[i] for i in top_ids]
        print(f"  {qw}: {nbrs}")

def analogy_test(embeddings, word2idx, idx2word, tests, top_k=3):
    """tests: list of (a, b, c) → b - a + c"""
    norms  = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms
    print("\n=== Analogy Tests ===")
    for a, b, c in tests:
        missing = [w for w in [a, b, c] if w not in word2idx]
        if missing:
            print(f"  {a}:{b}::{c}:? — missing {missing}")
            continue
        vec = (normed[word2idx[b]] - normed[word2idx[a]] + normed[word2idx[c]])
        sims = normed @ vec
        # exclude a, b, c
        excl = {word2idx[a], word2idx[b], word2idx[c]}
        ranked = [(sims[i], idx2word[i]) for i in np.argsort(sims)[::-1] if i not in excl]
        top_candidates = [w for _, w in ranked[:top_k]]
        print(f"  {a}:{b}::{c}:? → {top_candidates}")

def compute_mrr(embeddings, word2idx, idx2word, pairs):
    """pairs: list of (query, expected) – 20 manually defined."""
    norms  = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms
    rr_sum = 0.0
    found  = 0
    for query, expected in pairs:
        if query not in word2idx or expected not in word2idx:
            continue
        qi   = word2idx[query]
        sims = normed @ normed[qi]
        ranked = np.argsort(sims)[::-1]
        for rank, idx in enumerate(ranked, 1):
            if idx2word[idx] == expected:
                rr_sum += 1.0 / rank
                found  += 1
                break
    mrr = rr_sum / max(found, 1)
    return mrr


# ─────────────────────────────────────────────
# Four-Condition Comparison  (C1–C4)
# ─────────────────────────────────────────────
def condition_summary(label, embeddings, word2idx, idx2word, query_words, mrr_pairs):
    print(f"\n{'='*60}")
    print(f"Condition: {label}")
    top_k_neighbours(embeddings, word2idx, idx2word, query_words[:5], k=5)
    mrr = compute_mrr(embeddings, word2idx, idx2word, mrr_pairs)
    print(f"  MRR on 20 pairs: {mrr:.4f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load data ──
    cleaned_path = 'cleaned.txt'
    raw_path     = 'raw.txt'
    meta_path    = 'Metadata.json'

    # fall back to current dir sample if files absent
    if not os.path.exists(cleaned_path):
        print("cleaned.txt not found – creating tiny demo corpus")
        sample = "\n".join([f"[{i}]\nپاکستان حکومت وزیر اعظم پارلیمان الیکشن سیاسی\n"
                            for i in range(1, 51)])
        with open(cleaned_path, 'w', encoding='utf-8') as f: f.write(sample)
    if not os.path.exists(raw_path):
        with open(raw_path, 'w', encoding='utf-8') as f: f.write(open(cleaned_path).read())

    docs_clean = load_documents(cleaned_path)
    docs_raw   = load_documents(raw_path)

    all_tokens_clean = []
    for d in docs_clean:
        all_tokens_clean.extend(tokenize(d))

    word2idx, idx2word, counts = build_vocab(all_tokens_clean, max_vocab=10000)
    print(f"Vocabulary size: {len(word2idx)}")

    # ── 1.1 TF-IDF ──
    print("\n[1.1] Computing TF-IDF …")
    tfidf_matrix, idf = compute_tfidf(docs_clean, word2idx)
    np.save('tfidf_matrix.npy', tfidf_matrix)
    print(f"Saved tfidf_matrix.npy  shape={tfidf_matrix.shape}")
    top10_per_category(tfidf_matrix, docs_clean, word2idx, idx2word, meta_path)

    # ── 1.2 PPMI ──
    print("\n[1.2] Computing PPMI …")
    ppmi_matrix = compute_ppmi(docs_clean, word2idx, window=5)
    np.save('ppmi_matrix.npy', ppmi_matrix)
    print(f"Saved ppmi_matrix.npy  shape={ppmi_matrix.shape}")

    query_words_ppmi = ['پاکستان','حکومت','وزیر','کرکٹ','بجلی','کینسر','ایران','بلوچستان','سولر','میچ']
    nearest_neighbours_cosine(ppmi_matrix, word2idx, idx2word, query_words_ppmi, top_k=5)

    tsne_visualise(ppmi_matrix, idx2word, word2idx, counts, top_n=200)

    # ── 2.1 Skip-gram on cleaned.txt (C3) ──
    print("\n[2.1] Training Skip-gram on cleaned.txt …")
    emb_c3, model_c3 = train_skipgram(
        docs_clean, word2idx, idx2word, counts,
        embed_dim=100, window=5, K=10, epochs=5,
        batch_size=512, lr=0.001, device=device,
        save_path='embeddings_w2v.npy',
        loss_plot_path='w2v_loss_c3.png'
    )

    # ── 2.2 Evaluation ──
    query_words_w2v = ['Pakistan','Hukumat','Adalat','Maeeshat','Fauj',
                       'Sehat','Taleem','Aabadi',
                       'پاکستان','حکومت']
    top_k_neighbours(emb_c3, word2idx, idx2word, query_words_w2v, k=10)

    analogy_tests = [
        ('لاہور','پاکستان','دہلی'),
        ('وزیر','حکومت','صدر'),
        ('کرکٹ','میچ','فٹبال'),
        ('بیمار','ہسپتال','پڑھنا'),
        ('بجلی','سولر','پانی'),
        ('عورت','ماں','مرد'),
        ('ملک','پاکستان','شہر'),
        ('پارلیمان','الیکشن','عدالت'),
        ('استاد','تعلیم','ڈاکٹر'),
        ('بلوچستان','کوئٹہ','سندھ'),
    ]
    analogy_test(emb_c3, word2idx, idx2word, analogy_tests, top_k=3)

    # MRR pairs (manually defined)
    mrr_pairs = [
        ('پاکستان','حکومت'), ('وزیر','پارلیمان'), ('کرکٹ','میچ'),
        ('بجلی','سولر'),    ('کینسر','ہسپتال'),  ('ایران','امریکہ'),
        ('الیکشن','سیاسی'), ('بلوچستان','کوئٹہ'), ('تعلیم','اسکول'),
        ('مہنگائی','بجٹ'),  ('فوج','سیکیورٹی'),  ('عدالت','قانون'),
        ('تجارت','برآمد'), ('سیلاب','نقصان'),    ('صدر','وزیر'),
        ('اسکور','ٹیم'),   ('علاج','دوا'),       ('معاہدہ','سفارت'),
        ('بینک','قرض'),    ('اسمبلی','الیکشن'),
    ]

    # ── Four-Condition Comparison ──
    print("\n[Four-Condition Comparison]")

    # C1: PPMI
    condition_summary("C1 – PPMI baseline", ppmi_matrix, word2idx, idx2word,
                      ['پاکستان','حکومت','وزیر','کرکٹ','بجلی'], mrr_pairs)

    # C2: Skip-gram on raw.txt
    print("\nTraining C2 (raw.txt) …")
    all_tokens_raw = []
    for d in docs_raw:
        all_tokens_raw.extend(tokenize(d))
    word2idx_raw, idx2word_raw, counts_raw = build_vocab(all_tokens_raw, max_vocab=10000)
    emb_c2, _ = train_skipgram(
        docs_raw, word2idx_raw, idx2word_raw, counts_raw,
        embed_dim=100, window=5, K=10, epochs=5,
        batch_size=512, lr=0.001, device=device,
        save_path='embeddings_c2_raw.npy',
        loss_plot_path='w2v_loss_c2.png'
    )
    condition_summary("C2 – Skip-gram on raw.txt", emb_c2, word2idx_raw, idx2word_raw,
                      ['پاکستان','حکومت','وزیر','کرکٹ','بجلی'], mrr_pairs)

    # C3 already done
    condition_summary("C3 – Skip-gram on cleaned.txt", emb_c3, word2idx, idx2word,
                      ['پاکستان','حکومت','وزیر','کرکٹ','بجلی'], mrr_pairs)

    # C4: d=200
    print("\nTraining C4 (d=200) …")
    emb_c4, _ = train_skipgram(
        docs_clean, word2idx, idx2word, counts,
        embed_dim=200, window=5, K=10, epochs=5,
        batch_size=512, lr=0.001, device=device,
        save_path='embeddings_c4_d200.npy',
        loss_plot_path='w2v_loss_c4.png'
    )
    condition_summary("C4 – Skip-gram d=200", emb_c4, word2idx, idx2word,
                      ['پاکستان','حکومت','وزیر','کرکٹ','بجلی'], mrr_pairs)

    print("\n✓ Part 1 complete.")


if __name__ == '__main__':
    main()
