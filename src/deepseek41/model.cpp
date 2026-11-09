#include "strata/deepseek41/model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::deepseek41 {
namespace {
[[noreturn]] void fail(const std::string& s) { throw std::runtime_error("DeepSeek V4.1: " + s); }
constexpr const char* prefix = "deepseek41.";
std::string key(const char* s) { return std::string(prefix) + s; }
bool integral(MetaType t) {
    return t == MetaType::U8 || t == MetaType::I8 || t == MetaType::U16 || t == MetaType::I16 ||
           t == MetaType::U32 || t == MetaType::I32 || t == MetaType::U64 || t == MetaType::I64;
}
uint64_t integer(const GgufFile& f, const char* suffix, bool zero = false) {
    const auto* v = f.get(key(suffix));
    if (!v || !integral(v->type) || v->u > uint64_t(std::numeric_limits<int>::max()) || (!zero && !v->u))
        fail(key(suffix) + " must be a " + (zero ? "nonnegative" : "positive") + " integer");
    return v->u;
}
double real(const GgufFile& f, const char* suffix, bool zero = false) {
    const auto* v = f.get(key(suffix));
    if (!v || !v->is_num() || v->type == MetaType::BOOL || !std::isfinite(v->num()) ||
        (zero ? v->num() < 0 : v->num() <= 0)) fail(key(suffix) + " must be finite and positive");
    return v->num();
}
std::vector<double> per_layer(const GgufFile& f, const char* suffix, size_t n) {
    const auto* v = f.get(key(suffix));
    if (!v) fail("missing " + key(suffix));
    auto out = v->type == MetaType::ARRAY ? f.real_array(key(suffix)) : std::vector<double>(n, v->num());
    if (out.size() < n || (v->type != MetaType::ARRAY && (!v->is_num() || v->type == MetaType::BOOL)))
        fail(key(suffix) + " must provide one clamp per layer");
    out.resize(n);
    for (double x : out) if (!std::isfinite(x) || x <= 0) fail(key(suffix) + " has an invalid clamp");
    return out;
}
std::vector<uint64_t> integers(const GgufFile& f, const char* suffix) { return f.integer_array(key(suffix)); }
uint64_t integer_alias(const GgufFile& f, const char* primary, const char* alias, bool zero = false) {
    if (!f.get(key(primary))) return integer(f,alias,zero);
    const auto value = integer(f,primary,zero);
    if (f.get(key(alias)) && integer(f,alias,zero) != value) fail("conflicting " + key(primary) + " / " + key(alias));
    return value;
}
std::vector<uint64_t> integers_alias(const GgufFile& f, const char* primary, const char* alias) {
    if (!f.get(key(primary))) return integers(f,alias);
    auto value = integers(f,primary);
    if (f.get(key(alias)) && integers(f,alias) != value) fail("conflicting " + key(primary) + " / " + key(alias));
    return value;
}
void count(size_t actual, size_t expected, const char* what) {
    if (actual != expected) fail(std::string(what) + " length " + std::to_string(actual) +
                                 ", expected " + std::to_string(expected));
}
std::string block(int layer) { return "blk." + std::to_string(layer) + "."; }
}

Config Config::read(const GgufFile& f) {
    const auto* arch = f.get("general.architecture");
    if (!arch || arch->type != MetaType::STRING || arch->s != "deepseek41") fail("expected architecture deepseek41");
    Config c;
    c.layers = (int) integer(f, "block_count");
    c.hidden = (int) integer(f, "embedding_length");
    c.heads = (int) integer(f, "attention.head_count");
    c.kv_heads = (int) integer(f, "attention.head_count_kv");
    c.head_dim = (int) integer(f, "attention.key_length");
    if (integer(f, "attention.value_length") != uint64_t(c.head_dim)) fail("key/value head dimensions differ");
    c.rope_dim = (int) integer(f, "rope.dimension_count");
    c.q_rank = (int) integer(f, "attention.q_lora_rank");
    c.o_rank = (int) integer(f, "attention.output_lora_rank");
    c.o_groups = (int) integer(f, "attention.output_group_count");
    c.sliding_window = (int) integer(f, "attention.sliding_window");
    c.experts = (int) integer(f, "expert_count");
    c.top_k = (int) integer(f, "expert_used_count");
    c.expert_ff = (int) integer(f, "expert_feed_forward_length");
    c.shared_experts = (int) integer(f, "expert_shared_count");
    c.expert_scale = real(f, "expert_weights_scale");
    const auto* normalized = f.get(key("expert_weights_norm"));
    if (!normalized || normalized->type != MetaType::BOOL || !normalized->u || integer(f, "expert_gating_func") != 4)
        fail("requires normalized sqrtsoftplus expert routing");
    if (const auto* hash = f.get(key("hash_layer_count")); hash && hash->u != 0) fail("hash routing is not V4.1 routing");
    c.hc = (int) integer(f, "hyper_connection.count");
    c.sinkhorn_iterations = (int) integer(f, "hyper_connection.sinkhorn_iterations");
    c.hc_eps = real(f, "hyper_connection.epsilon");
    c.rms_eps = real(f, "attention.layer_norm_rms_epsilon");
    c.index_heads = (int) integer(f, "attention.indexer.head_count");
    c.index_dim = (int) integer(f, "attention.indexer.key_length");
    c.index_top_k = (int) integer(f, "attention.indexer.top_k");
    c.context_train = (int) integer(f, "context_length");
    c.rope_context_original = (int) integer(f, "rope.scaling.original_context_length");
    c.rope_base = real(f, "rope.freq_base");
    c.compress_rope_base = real(f, "attention.compress_rope_freq_base");
    c.rope_scale = real(f, "rope.scaling.factor");
    c.beta_fast = real(f, "rope.scaling.yarn_beta_fast");
    c.beta_slow = real(f, "rope.scaling.yarn_beta_slow");
    const auto* scaling = f.get(key("rope.scaling.type"));
    if (!scaling || scaling->s != "yarn") fail("requires YaRN compressed-stream RoPE");
    const auto* tokens = f.get("tokenizer.ggml.tokens");
    if (!tokens || tokens->type != MetaType::ARRAY || tokens->elem != MetaType::STRING ||
        !tokens->count || tokens->count > uint64_t(std::numeric_limits<int>::max())) fail("invalid vocabulary");
    c.vocab = (int) tokens->count;
    if (c.layers > 1024 || c.hc > 16 || c.kv_heads != 1 || c.heads % c.o_groups ||
        c.top_k > c.experts || c.top_k > 64 || c.rope_dim % 2 || c.rope_dim > c.head_dim || c.rope_dim > c.index_dim)
        fail("incompatible attention, routing or hyper-connection dimensions");
    // CUDA launch dimensions and strides are signed32-bit, including flattened HC/output projections.
    for (const uint64_t n : {uint64_t(c.hidden)*(c.hc+1), uint64_t(c.hc)*(c.hc+2),
                           uint64_t(c.heads)*c.head_dim, uint64_t(c.o_rank)*c.o_groups,
                           uint64_t(c.index_heads)*c.index_dim, uint64_t(c.expert_ff)*c.shared_experts})
        if (n > uint64_t(std::numeric_limits<int>::max())) fail("flattened dimension overflows int32");
    c.compress_ratios = integers(f, "attention.compress_ratios");
    if (c.compress_ratios.size() < size_t(c.layers)) fail("compress_ratios is shorter than block_count");
    c.compress_ratios.resize(c.layers);
    for (auto n : c.compress_ratios) if (n > 2) fail("V4.1 compress ratio must be0,1,2");
    c.clamp_expert = per_layer(f, "swiglu_clamp_exp", c.layers);
    c.clamp_shared = f.get(key("swiglu_clamp_shexp")) ? per_layer(f, "swiglu_clamp_shexp", c.layers) : c.clamp_expert;
    auto& e = c.engram;
    const bool global = f.get(key("ple.layers")) != nullptr;
    e.heads = (int) integer(f, global ? "ple.heads_per_ngram" : "engram.head_count");
    e.head_dim = (int) (global ? integer(f,"embedding_length_per_layer_input") : integer_alias(f, "engram.key_length", "engram.head_dim"));
    e.max_ngram = (int) integer(f, global ? "ple.ngram_size" : "engram.max_ngram_size");
    e.pad_id = integer_alias(f, "engram.pad_id", "engram.pad_token_id", true);
    if (e.heads > 1024 || e.max_ngram < 2 || e.max_ngram > 128) fail("invalid Engram bucket geometry");
    e.layers = global ? integers(f,"ple.layers") : integers_alias(f, "engram.layer_ids", "engram.layers");
    e.multipliers = integers(f,global ? "ple.layer_multipliers" : "engram.multipliers");
    e.primes = integers(f,global ? "ple.head_vocab_sizes" : "engram.primes");
    e.offsets = integers(f,global ? "ple.head_offsets" : "engram.offsets");
    e.token_map = integers(f, "engram.token_map");
    if (global) {
        // PLE pad_token_id names a vocabulary token; hashes use its mapped ID.
        if (e.pad_id >= e.token_map.size()) fail("Engram pad token is outside the vocabulary");
        e.pad_id=e.token_map[e.pad_id];
        for (const char* old : {"engram.head_count","engram.key_length","engram.head_dim",
                               "engram.max_ngram_size","engram.layer_ids","engram.layers",
                               "engram.multipliers","engram.primes","engram.offsets"})
            if (f.get(key(old))) fail("mixed PLE and legacy Engram metadata");
    }
    if (e.layers.empty() || e.layers.size() > size_t(c.layers)) fail("invalid Engram layer count");
    count(e.multipliers.size(), e.layers.size()*size_t(e.max_ngram), "Engram multipliers");
    count(e.primes.size(), e.layers.size()*e.buckets(), "Engram primes");
    count(e.offsets.size(), e.primes.size(), "Engram offsets"); count(e.token_map.size(), size_t(c.vocab), "Engram token_map");
    std::set<uint64_t> seen;
    for (uint64_t layer : e.layers) if (layer >= uint64_t(c.layers) || !seen.insert(layer).second) fail("invalid/duplicate Engram layer");
    for (size_t i = 0; i < e.primes.size(); ++i)
        if (!e.primes[i] || e.offsets[i] >= uint64_t(INT32_MAX) || e.primes[i] > uint64_t(INT32_MAX) - e.offsets[i])
            fail("Engram bucket exceeds int32 row range");
    if (uint64_t(e.head_dim)*e.buckets() > uint64_t(INT32_MAX)) fail("Engram projection dimension overflows");
    if (global) {
        for (size_t l=0;l<e.layers.size();++l) {
            auto first=e.offsets.begin()+l*e.buckets(), last=first+e.buckets();
            const uint64_t base=*std::min_element(first,last);
            e.table_offsets.push_back(base);
            for (auto it=first;it!=last;++it) *it-=base;
        }
    }
    return c;
}

Model::Model(const std::string& any_shard) : gguf_(gguf_split_paths(any_shard)), config_(Config::read(gguf_.meta())) { make_engram_views(); validate(); }

TensorRef Model::tensor(const std::string& name) const {
    size_t shard = 0;
    const auto* t = gguf_.find(name, &shard);
    const auto view=engram_views_.find(name);
    if (!t && view==engram_views_.end()) fail("missing tensor " + name);
    const TensorRef ref=t ? TensorRef{&gguf_.shard(shard),t} : TensorRef{view->second.file,&view->second.tensor};
    const uint64_t bytes = ref.bytes(), payload = ref.file->file_size()-ref.file->data_start();
    if (!bytes || ref.tensor->offset > payload || bytes > payload-ref.tensor->offset) fail("truncated/unsupported tensor " + name);
    return ref;
}


void Model::make_engram_views() {
    const auto& e=config_.engram;
    if (e.table_offsets.empty()) return;
    const auto source=tensor("per_layer_token_embd.weight");
    if (source.tensor->shape.size()!=2 || source.tensor->shape[0]!=uint64_t(e.head_dim))
        fail("invalid global Engram table shape");
    const uint64_t stride=engram_row_bytes(source.tensor->type,e.head_dim);
    if (!stride) fail("unsupported global Engram table type");
    std::vector<std::pair<uint64_t,uint64_t>> spans;
    for (size_t l=0;l<e.layers.size();++l) {
        uint64_t rows=0;
        for (size_t k=0;k<e.buckets();++k) {
            const size_t i=l*e.buckets()+k;
            rows=std::max(rows,e.offsets[i]+e.primes[i]);
        }
        const uint64_t base=e.table_offsets[l];
        if (base>source.tensor->shape[1] || rows>source.tensor->shape[1]-base ||
            base>UINT64_MAX/stride || source.tensor->offset>UINT64_MAX-base*stride)
            fail("Engram view exceeds global table");
        const auto name=block(int(e.layers[l]))+"engram_embd.weight";
        if (gguf_.find(name)) fail("mixed global and per-layer Engram tables");
        auto t=*source.tensor;t.name=name;t.shape[1]=rows;t.offset+=base*stride;
        engram_views_.emplace(name,View{source.file,std::move(t)});
        spans.emplace_back(base,base+rows);
    }
    std::sort(spans.begin(),spans.end());
    for(size_t i=1;i<spans.size();++i)if(spans[i].first<spans[i-1].second)fail("overlapping Engram layer views");
}

void Model::require(const std::string& name, const std::vector<uint64_t>& shape, int group,
                    const std::vector<uint64_t>& alternative) {
    const auto ref = tensor(name);
    if (ref.tensor->shape != shape && (alternative.empty() || ref.tensor->shape != alternative)) fail("wrong shape: " + name);
    const auto type = ref.tensor->type;
    if (type != 0 && type != 1 && type != 8 && type != 12 && type != 13 && type != 14 && type != 30 &&
        !(group == 1 && (type == 16 || type == 17 || type == 19 || type == 22 || type == 29 || type == 39)) &&
        !(group == 2 && (type == 20 || type == fp8_engram_type || type == mxfp8_engram_type)))
        fail("unsupported type " + std::string(ref.tensor->type_name()) + ": " + name);
    // The physical global table is counted/validated once, not once per view.
    if (engram_views_.count(name)) return;
    if (!checked_.insert(name).second) fail("duplicate layout entry " + name);
    auto& bytes = group == 1 ? routed_bytes_ : (group == 2 ? engram_bytes_ : dense_bytes_);
    if (ref.bytes() > UINT64_MAX - bytes) fail("tensor byte count overflow");
    bytes += ref.bytes();
}

void Model::validate() {
    const auto& c = config_;
    if (!c.engram.table_offsets.empty()) {
        const auto t=tensor("per_layer_token_embd.weight");
        require("per_layer_token_embd.weight",t.tensor->shape,2);
    }
    const uint64_t H = c.hidden, V = c.vocab, A = c.heads, D = c.head_dim, Q = c.q_rank;
    const uint64_t G = c.o_groups, R = c.o_rank, HC = c.hc, E = c.experts, F = c.expert_ff;
    const uint64_t I = c.index_dim, N = c.index_heads, C = HC*(HC+2);
    require("token_embd.weight", {H,V}); require("output_norm.weight", {H}); require("output.weight", {H,V});
    streams_.resize(c.layers);
    int last_kv = -1, last_key = -1, last_topk = -1;
    for (int l = 0; l < c.layers; ++l) {
        const auto b = block(l);
        auto need = [&](const char* suffix, const std::vector<uint64_t>& shape, int group = 0) { require(b+suffix,shape,group); };
        auto has = [&](const char* suffix) { return gguf_.find(b+suffix) != nullptr; };
        need("attn_norm.weight",{H}); need("attn_sinks.weight",{A});
        need("attn_q_a.weight",{H,Q}); need("attn_q_a_norm.weight",{Q}); need("attn_q_b.weight",{Q,A*D});
        need("attn_kv.weight",{H,D}); need("attn_kv_a_norm.weight",{D});
        require(b+"attn_output_a.weight",{A*D/G,R*G},0,{A*D/G,R,G}); need("attn_output_b.weight",{R*G,H});
        for (const auto* sublayer : {"attn","ffn"}) {
            const auto h = "hc_" + std::string(sublayer);
            require(b+h+"_fn.weight",{H*HC,C}); require(b+h+"_base.weight",{C}); require(b+h+"_scale.weight",{3});
        }
        need("ffn_norm.weight",{H}); need("ffn_gate_inp.weight",{H,E}); need("exp_probs_b.bias",{E});
        if (has("exp_probs_b_vl.bias")) need("exp_probs_b_vl.bias",{E});
        need("ffn_gate_exps.weight",{H,F,E},1); need("ffn_up_exps.weight",{H,F,E},1); need("ffn_down_exps.weight",{F,H,E},1);
        need("ffn_gate_shexp.weight",{H,F*uint64_t(c.shared_experts)});
        need("ffn_up_shexp.weight",{H,F*uint64_t(c.shared_experts)});
        need("ffn_down_shexp.weight",{F*uint64_t(c.shared_experts),H});
        auto& s = streams_[l];
        s.publishes_kv = has("attn_compressor_kv.weight");
        s.publishes_keys = has("indexer.attn_k.weight");
        s.publishes_top_k = has("indexer.attn_q_b.weight");
        const auto ratio = c.compress_ratios[l];
        if (s.publishes_kv) {
            if (!ratio) fail(b+"compressor on pure sliding-window layer");
            need("attn_compressor_kv.weight",{H,D}); need("attn_compressor_norm.weight",{D});
            if (ratio > 1 && !has("attn_compressor_gate.weight")) fail(b+"pooling compressor has no gate");
            if (has("attn_compressor_gate.weight")) need("attn_compressor_gate.weight",{H,D});
            last_kv = l;
        } else if (has("attn_compressor_norm.weight") || has("attn_compressor_gate.weight")) fail(b+"orphan compressor weights");
        if (s.publishes_keys) {
            if (!s.publishes_kv) fail(b+"index keys have no KV source");
            need("indexer.attn_k.weight",{D,I}); need("indexer.k_norm.weight",{I}); last_key = l;
        } else if (has("indexer.k_norm.weight")) fail(b+"orphan index key norm");
        if (s.publishes_top_k) {
            need("indexer.attn_q_b.weight",{Q,N*I}); need("indexer.proj.weight",{H,N}); last_topk = l;
        } else if (has("indexer.proj.weight")) fail(b+"orphan index query projection");
        if (ratio) {
            if (last_kv < 0 || last_key < 0 || last_topk < 0) fail(b+"reads compressed state before a source publishes it");
            if (c.compress_ratios[last_kv] != ratio) fail(b+"compression ratio disagrees with KV source");
            s.kv_source = last_kv; s.key_source = last_key; s.top_k_source = last_topk;
        } else if (s.publishes_keys || s.publishes_top_k) fail(b+"indexer on pure sliding-window layer");
        const auto& ec = c.engram;
        const auto it = std::find(ec.layers.begin(),ec.layers.end(),uint64_t(l));
        if (it != ec.layers.end()) {
            s.engram_index = (int) (it-ec.layers.begin());
            const auto table = tensor(b+"engram_embd.weight");
            if (table.tensor->shape.size()!=2 || table.tensor->shape[0]!=uint64_t(ec.head_dim)) fail(b+"invalid Engram table shape");
            uint64_t max_row = 0;
            for (size_t k = 0; k < ec.buckets(); ++k) {
                const size_t at = size_t(s.engram_index)*ec.buckets()+k;
                max_row = std::max(max_row,ec.offsets[at]+ec.primes[at]);
            }
            if (max_row > table.tensor->shape[1]) fail(b+"Engram buckets exceed table rows");
            require(b+"engram_embd.weight",table.tensor->shape,2);
            const bool wkv = has("engram_wkv.weight"), kv = has("engram_kv.weight");
            if (wkv == kv) fail(b+"requires exactly one Engram KV projection");
            s.engram_projection = b+(wkv ? "engram_wkv.weight" : "engram_kv.weight");
            require(s.engram_projection,{ec.buckets()*uint64_t(ec.head_dim),H*(HC+1)});
            s.engram_fused_gate = has("engram_gate.weight");
            if (s.engram_fused_gate) {
                if (has("engram_q.weight") || has("engram_k.weight")) fail(b+"mixed fused and separate Engram gates");
                need("engram_gate.weight",{H,HC});
            } else {
                need("engram_q.weight",{H,HC}); need("engram_k.weight",{H,HC});
            }
        }
    }
    for (size_t shard = 0; shard < gguf_.size(); ++shard)
        for (const auto& t : gguf_.shard(shard).tensors())
            if (!checked_.count(t.name)) fail("tensor is not part of the native V4.1 layout: "+t.name);
}

std::vector<int32_t> engram_rows(const EngramConfig& c, size_t engram, int32_t token,
                                const std::vector<int32_t>& history) {
    if (c.heads <= 0 || c.max_ngram < 2 || engram >= c.layers.size() ||
        c.multipliers.size() != c.layers.size()*size_t(c.max_ngram) ||
        c.primes.size() != c.layers.size()*c.buckets() || c.offsets.size() != c.primes.size()) fail("invalid Engram hash configuration");
    auto mapped = [&](int32_t id) { return id < 0 || uint64_t(id) >= c.token_map.size() ? c.pad_id : c.token_map[size_t(id)]; };
    const auto* mult = c.multipliers.data()+engram*size_t(c.max_ngram);
    uint64_t rolling = mapped(token)*mult[0];
    bool blocked = false;
    std::vector<int32_t> result(c.buckets());
    for (int s = 1; s < c.max_ngram; ++s) {
        const int32_t prior = size_t(s) <= history.size() ? history[history.size()-size_t(s)] : -1;
        blocked = blocked || prior < 0;
        rolling ^= (blocked ? c.pad_id : mapped(prior))*mult[s];
        for (int h = 0; h < c.heads; ++h) {
            const size_t bucket = size_t(s-1)*size_t(c.heads)+size_t(h), at = engram*c.buckets()+bucket;
            if (!c.primes[at] || c.offsets[at] >= uint64_t(INT32_MAX) || c.primes[at] > uint64_t(INT32_MAX)-c.offsets[at])
                fail("invalid Engram hash bucket");
            result[bucket] = (int32_t) (rolling%c.primes[at]+c.offsets[at]);
        }
    }
    return result;
}
}  // namespace strata::deepseek41
