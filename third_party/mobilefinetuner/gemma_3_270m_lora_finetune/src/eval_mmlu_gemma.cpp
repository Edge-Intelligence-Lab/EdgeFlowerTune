#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <filesystem>

#include "finetune_ops/graph/gemma_model.h"
#include "finetune_ops/graph/gemma_lora_injector.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/core/tokenizer_hf.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/memory_manager.h"

using namespace std;
using namespace ops;
namespace fs = std::filesystem;

struct Args {
    string mmlu_root = "data/mmlu/data";
    string split = "dev"; // dev|test
    string pretrained_dir = "pretrained";
    string lora_path;
    bool lora_merge = true; // placeholder flag if LoRA merge is added later
    int fewshot = 0;
    string out_file;
    string tokenizer_json;
    int max_seq_len = 0; // 0 = no truncation
};

static void usage(const char* prog) {
    cerr << "Usage: " << prog <<
        " --mmlu_root PATH [--split dev|test] [--fewshot K] [--pretrained_dir PATH]" << endl;
    cerr << "  [--lora_path PATH]" << endl;
    cerr << "  [--max_seq_len N]" << endl;
    cerr << "  [--out FILE]" << endl;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        string k = argv[i];
        auto get = [&](const string& key){ if (i+1>=argc || k!=key) { usage(argv[0]); exit(1);} return string(argv[++i]); };
        if (k == "--mmlu_root") a.mmlu_root = get("--mmlu_root");
        else if (k == "--split") a.split = get("--split");
        else if (k == "--fewshot") a.fewshot = stoi(get("--fewshot"));
        else if (k == "--pretrained_dir") a.pretrained_dir = get("--pretrained_dir");
        else if (k == "--tokenizer_json") a.tokenizer_json = get("--tokenizer_json");
        else if (k == "--out") a.out_file = get("--out");
        else if (k == "--lora_path") a.lora_path = get("--lora_path");
        else if (k == "--max_seq_len") a.max_seq_len = stoi(get("--max_seq_len"));
        else if (k == "--help" || k == "-h") { usage(argv[0]); exit(0);} 
        else { cerr << "Unknown arg: " << k << endl; usage(argv[0]); exit(1); }
    }
    if (a.split != "dev" && a.split != "test") {
        cerr << "Invalid --split: " << a.split << ", must be dev|test" << endl;
        exit(1);
    }
    return a;
}

static inline string trim_copy(const string& s) {
    size_t l = 0, r = s.size();
    while (l < r && isspace(static_cast<unsigned char>(s[l]))) ++l;
    while (r > l && isspace(static_cast<unsigned char>(s[r-1]))) --r;
    return s.substr(l, r - l);
}

static vector<string> parse_csv_line(const string& line) {
    vector<string> fields;
    string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i+1] == '"') {
                    cur.push_back('"'); ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push_back(c);
            }
        } else {
            if (c == ',') { fields.emplace_back(move(cur)); cur.clear(); }
            else if (c == '"') { in_quotes = true; }
            else { cur.push_back(c); }
        }
    }
    fields.emplace_back(move(cur));
    return fields;
}

struct MCQItem {
    string subject;
    string question;
    string A, B, C, D;
    char answer;
};

static void read_mmlu_csv(const string& path, vector<MCQItem>& out_items) {
    ifstream in(path);
    if (!in) return;
    string first;
    if (!getline(in, first)) return;
    auto cols = parse_csv_line(first);
    auto find_col = [&](const string& name)->int{
        for (size_t i = 0; i < cols.size(); ++i) {
            string c = trim_copy(cols[i]);
            transform(c.begin(), c.end(), c.begin(), ::tolower);
            if (c == name) return static_cast<int>(i);
        }
        return -1;
    };
    int idx_subject = find_col("subject");
    int idx_question = find_col("question");
    int idx_a = find_col("a");
    int idx_b = find_col("b");
    int idx_c = find_col("c");
    int idx_d = find_col("d");
    int idx_answer = find_col("answer");
    bool has_header = (idx_question >= 0 && idx_a >= 0 && idx_b >= 0 && idx_c >= 0 && idx_d >= 0 && idx_answer >= 0);
    string file_subject = fs::path(path).stem().string();

    auto handle_row = [&](const vector<string>& f) {
        if (f.size() < 6) return;
        MCQItem item;
        if (has_header && idx_subject >= 0) {
            item.subject = trim_copy(f[idx_subject]);
        } else {
            item.subject = file_subject;
        }
        int q_idx = has_header ? idx_question : 0;
        int a_idx = has_header ? idx_a : 1;
        int b_idx = has_header ? idx_b : 2;
        int c_idx = has_header ? idx_c : 3;
        int d_idx = has_header ? idx_d : 4;
        int ans_idx = has_header ? idx_answer : 5;
        if (q_idx < 0 || a_idx < 0 || b_idx < 0 || c_idx < 0 || d_idx < 0 || ans_idx < 0) return;
        if (static_cast<int>(f.size()) <= max({q_idx, a_idx, b_idx, c_idx, d_idx, ans_idx})) return;
        item.question = trim_copy(f[q_idx]);
        item.A = trim_copy(f[a_idx]);
        item.B = trim_copy(f[b_idx]);
        item.C = trim_copy(f[c_idx]);
        item.D = trim_copy(f[d_idx]);
        string ans = trim_copy(f[ans_idx]);
        item.answer = ans.empty() ? 'A' : static_cast<char>(toupper(static_cast<unsigned char>(ans[0])));
        out_items.emplace_back(move(item));
    };

    if (!has_header) {
        if (!trim_copy(first).empty()) handle_row(cols);
    }
    string line;
    while (getline(in, line)) {
        if (trim_copy(line).empty()) continue;
        auto f = parse_csv_line(line);
        handle_row(f);
    }
}

static string build_prompt(const MCQItem& q, const vector<MCQItem>* shots) {
    auto one = [](const MCQItem& x) {
        return string("Question: ") + x.question + "\n"
             + "A. " + x.A + "\n"
             + "B. " + x.B + "\n"
             + "C. " + x.C + "\n"
             + "D. " + x.D + "\n"
             + "Answer: ";
    };
    string prompt;
    if (shots && !shots->empty()) {
        for (const auto& s : *shots) {
            prompt += one(s);
            prompt += s.answer;
            prompt += "\n\n";
        }
    }
    prompt += one(q);
    return prompt;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args = parse_args(argc, argv);
    try {
        cout << "========== MMLU Evaluation (Gemma) ==========" << endl;
        cout << "mmlu_root     : " << args.mmlu_root << endl;
        cout << "split         : " << args.split << endl;
        cout << "fewshot       : " << args.fewshot << endl;
        cout << "pretrained_dir: " << args.pretrained_dir << endl;
        if (!args.lora_path.empty()) {
            cout << "lora_path    : " << args.lora_path << endl;
        }

        auto cfg = GemmaTextConfig::from_pretrained(args.pretrained_dir);
        GemmaModel model(cfg);
        SafeTensorsReader reader(args.pretrained_dir + "/model.safetensors");
        reader.parse_header();
        auto mapping = GemmaKeyMapper::generate_gemma_mapping(cfg.num_hidden_layers);
        SafeTensorsLoadOptions load_opts;
        load_opts.verbose = false;
        auto tensors = reader.load_tensors_mapped(mapping, load_opts);
        for (auto& kv : tensors) model.assign_weight(kv.first, kv.second);

        if (!args.lora_path.empty()) {
            GemmaLoraSpec spec = GemmaLoraSpec::full_attn_mlp();
            GemmaLoraInjector injector;
            injector.inject(model, spec);
            injector.load_lora_safetensors(args.lora_path);
            injector.print_info();
            auto lora_params = model.get_lora_parameters();
            for (auto& p : lora_params) {
                if (p) p->set_requires_grad(false);
            }
        }

        const bool use_model_dir = args.tokenizer_json.empty();
        const std::string tok_path = use_model_dir ? args.pretrained_dir : args.tokenizer_json;
        HFTokenizer tok(tok_path, use_model_dir);
        if (!tok.load()) {
            throw std::runtime_error(
                "Failed to load HF tokenizer natively. Native HF tokenizer is now the standard path; "
                "rebuild with -DUSE_HF_TOKENIZERS=ON.");
        }
        int pad_id = tok.token_to_id("<pad>");
        if (pad_id < 0) pad_id = tok.token_to_id("<eos>");
        if (pad_id < 0) pad_id = tok.token_to_id("</s>");
        if (pad_id < 0) pad_id = 0;

        unordered_map<string, vector<MCQItem>> subj2items;
        string split_dir = args.mmlu_root + "/" + args.split;
        for (auto& p : fs::directory_iterator(split_dir)) {
            if (!p.is_regular_file() || p.path().extension() != ".csv") continue;
            vector<MCQItem> items;
            read_mmlu_csv(p.path().string(), items);
            for (auto& it : items) subj2items[it.subject].push_back(move(it));
        }
        cout << "[Eval] Loaded " << subj2items.size() << " subjects" << endl;

        auto encode_ids = [&](const string& text)->vector<int32_t>{
            return tok.encode(text);
        };

        auto last_letter = [&](const string& prompt)->char{
            auto ids32 = encode_ids(prompt);
            if (ids32.empty()) ids32.push_back(pad_id >= 0 ? pad_id : 0);
            if (args.max_seq_len > 0 && static_cast<int>(ids32.size()) > args.max_seq_len) {
                ids32 = vector<int32_t>(ids32.end() - args.max_seq_len, ids32.end());
            }
            vector<float> attn(ids32.size(), 1.0f);
            TensorPtr input_ids = make_shared<Tensor>(vector<int64_t>{1,(int64_t)ids32.size()}, ids32.data(), kInt32, kCPU);
            TensorPtr attention = make_shared<Tensor>(vector<int64_t>{1,(int64_t)ids32.size()}, attn.data(), kFloat32, kCPU);
            auto logits = model.forward(input_ids, attention);
            auto logits2d = flatten(logits, 0, 1);
            int64_t S = logits->shape()[1];
            int64_t V = logits2d->shape()[1];
            const float* all = logits2d->data<float>();
            vector<float> last_row(V);
            const float* src = all + (S - 1) * V;
            copy(src, src + V, last_row.begin());
            TensorPtr last_logits = make_shared<Tensor>(vector<int64_t>{1,V}, last_row.data(), kFloat32, kCPU);
            auto logp = log_softmax(last_logits, 1);
            const float* lp = logp->data<float>();
            // Leading space tends to be correct for SP-based models
            auto idA = encode_ids(" A"); int idxA = idA.empty()? encode_ids("A").front(): idA.back();
            auto idB = encode_ids(" B"); int idxB = idB.empty()? encode_ids("B").front(): idB.back();
            auto idC = encode_ids(" C"); int idxC = idC.empty()? encode_ids("C").front(): idC.back();
            auto idD = encode_ids(" D"); int idxD = idD.empty()? encode_ids("D").front(): idD.back();
            auto get_lp = [&](int idx)->float{ return (idx>=0 && idx < V) ? lp[idx] : -1e30f; };
            float sA = get_lp(idxA), sB = get_lp(idxB), sC = get_lp(idxC), sD = get_lp(idxD);
            char pred = 'A'; float best = sA;
            if (sB > best) { best = sB; pred = 'B'; }
            if (sC > best) { best = sC; pred = 'C'; }
            if (sD > best) { best = sD; pred = 'D'; }
            return pred;
        };

        struct Report { string subject; int correct=0; int total=0; };
        vector<Report> reports;
        int total_correct=0, total_count=0;
        for (auto& kv : subj2items) {
            const string& subj = kv.first;
            auto& items = kv.second;
            if (items.empty()) continue;
            int correct=0, count=0;
            vector<MCQItem> shots;
            if (args.fewshot > 0) {
                for (size_t i=0; i < (size_t)args.fewshot && i<items.size(); ++i) shots.push_back(items[i]);
            }
            for (size_t i=0; i<items.size(); ++i) {
                const auto& x = items[i];
                vector<MCQItem> shots_ex;
                if (args.fewshot > 0) {
                    shots_ex.reserve(shots.size());
                    for (const auto& s : shots) if (&s != &x) shots_ex.push_back(s);
                }
                auto prompt = build_prompt(x, args.fewshot > 0 ? &shots_ex : nullptr);
                char pred = last_letter(prompt);
                if (pred == x.answer) correct++;
                count++;
                MemoryManager::instance().force_cleanup();
            }
            reports.push_back({subj, correct, count});
            total_correct += correct; total_count += count;
        }
        sort(reports.begin(), reports.end(), [](const Report&a, const Report&b){ return a.subject < b.subject; });
        float macro = 0.0f;
        for (auto& r : reports) macro += (r.total>0 ? float(r.correct)/float(r.total) : 0.0f);
        if (!reports.empty()) macro /= float(reports.size());
        float micro = (total_count>0) ? float(total_correct)/float(total_count) : 0.0f;

        cout << "Per-subject:" << endl;
        for (auto& r : reports) {
            printf("  %-30s | n=%4d | acc=%.2f%%\n", r.subject.c_str(), r.total, (r.total>0? 100.0f*float(r.correct)/float(r.total) : 0.0f));
        }
        printf("\nMacro=%.2f%% | Micro=%.2f%%\n", 100.0f*macro, 100.0f*micro);
        if (!args.out_file.empty()) {
            ofstream out(args.out_file, ios::app);
            if (out) {
                for (auto& r : reports) {
                    out << "{\"task\":\"mmlu\",\"subject\":\"" << r.subject
                        << "\",\"n\":" << r.total << ",\"acc\":" << (r.total>0? float(r.correct)/float(r.total):0.0f) << "}\n";
                }
                out << "{\"task\":\"mmlu\",\"macro\":" << macro << ",\"micro\":" << micro
                    << ",\"split\":\"" << args.split << "\",\"fewshot\":" << args.fewshot << "}\n";
                out.close();
            }
        }
        cout << "\n✅ Done." << endl;
        return 0;
    } catch (const exception& e) {
        cerr << "\n❌ Exception: " << e.what() << endl;
        return 1;
    }
}
