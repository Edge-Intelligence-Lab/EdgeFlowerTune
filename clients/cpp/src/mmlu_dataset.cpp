#include "lshaped/mmlu_dataset.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lshaped {

namespace {

std::vector<std::string> ParseCsvRow(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool in_quotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (ch == ',' && !in_quotes) {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    fields.push_back(current);
    for (std::string& field : fields) {
        while (!field.empty() && (field.back() == '\r' || field.back() == '\n')) {
            field.pop_back();
        }
    }
    return fields;
}

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

int FindColumn(const std::vector<std::string>& headers, const std::vector<std::string>& candidates) {
    for (std::size_t i = 0; i < headers.size(); ++i) {
        const std::string normalized = Lower(headers[i]);
        for (const std::string& candidate : candidates) {
            if (normalized == candidate) {
                return static_cast<int>(i);
            }
        }
    }
    return -1;
}

std::vector<MMLUSample> LoadCsvSamples(const ClientOptions& options) {
    std::ifstream input(options.dataset_csv);
    if (!input) {
        throw std::runtime_error("Failed to open dataset_csv: " + options.dataset_csv);
    }

    std::string header_line;
    if (!std::getline(input, header_line)) {
        throw std::runtime_error("dataset_csv is empty: " + options.dataset_csv);
    }
    const std::vector<std::string> headers = ParseCsvRow(header_line);

    const int question_col = FindColumn(headers, {"question"});
    const int a_col = FindColumn(headers, {"a", "option_a"});
    const int b_col = FindColumn(headers, {"b", "option_b"});
    const int c_col = FindColumn(headers, {"c", "option_c"});
    const int d_col = FindColumn(headers, {"d", "option_d"});
    const int answer_col = FindColumn(headers, {"answer", "label"});
    if (question_col < 0 || a_col < 0 || b_col < 0 || c_col < 0 || d_col < 0 || answer_col < 0) {
        throw std::runtime_error("dataset_csv must include columns: question,A,B,C,D,answer");
    }

    std::vector<MMLUSample> samples;
    std::string line;
    std::size_t global_index = 0;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = ParseCsvRow(line);
        const int max_required = std::max({question_col, a_col, b_col, c_col, d_col, answer_col});
        if (static_cast<int>(fields.size()) <= max_required) {
            continue;
        }
        if (static_cast<int>(global_index % static_cast<std::size_t>(options.num_clients)) != options.client_index) {
            ++global_index;
            continue;
        }
        MMLUSample sample;
        sample.question = fields[question_col];
        sample.option_a = fields[a_col];
        sample.option_b = fields[b_col];
        sample.option_c = fields[c_col];
        sample.option_d = fields[d_col];
        const std::string answer_text = fields[answer_col];
        if (answer_text.empty()) {
            throw std::runtime_error("Empty answer column in dataset_csv");
        }
        sample.answer = static_cast<char>(std::toupper(answer_text[0]));
        if (sample.answer < 'A' || sample.answer > 'D') {
            throw std::runtime_error("Answer must be one of A/B/C/D");
        }
        samples.push_back(sample);
        ++global_index;
    }
    return samples;
}

std::vector<MMLUSample> MakeSyntheticSamples(const ClientOptions& options) {
    std::vector<MMLUSample> samples;
    samples.reserve(static_cast<std::size_t>(options.synthetic_samples));
    for (int i = 0; i < options.synthetic_samples; ++i) {
        const char answer = static_cast<char>('A' + (i % 4));
        MMLUSample sample;
        sample.question = "Synthetic question " + std::to_string(i) + " for client " + std::to_string(options.client_index);
        sample.option_a = "Option A " + std::to_string(i);
        sample.option_b = "Option B " + std::to_string(i);
        sample.option_c = "Option C " + std::to_string(i);
        sample.option_d = "Option D " + std::to_string(i);
        sample.answer = answer;
        samples.push_back(sample);
    }
    return samples;
}

}  // namespace

std::string MMLUSample::Prompt() const {
    std::ostringstream builder;
    builder << "Question: " << question << "\n";
    builder << "A. " << option_a << "\n";
    builder << "B. " << option_b << "\n";
    builder << "C. " << option_c << "\n";
    builder << "D. " << option_d << "\n";
    builder << "Answer: ";
    return builder.str();
}

std::string MMLUSample::AnswerLabel() const {
    return std::string(1, answer);
}

ClientShardDataset::ClientShardDataset(std::vector<MMLUSample> samples)
    : samples_(std::move(samples)) {
    if (samples_.empty()) {
        throw std::runtime_error("ClientShardDataset must not be empty");
    }
}

ClientShardDataset ClientShardDataset::FromOptions(const ClientOptions& options) {
    if (!options.dataset_csv.empty()) {
        return ClientShardDataset(LoadCsvSamples(options));
    }
    return ClientShardDataset(MakeSyntheticSamples(options));
}

std::vector<MMLUSample> ClientShardDataset::NextBatch(int batch_size) {
    assert(batch_size > 0);
    std::vector<MMLUSample> batch;
    batch.reserve(static_cast<std::size_t>(batch_size));
    for (int i = 0; i < batch_size; ++i) {
        batch.push_back(samples_[cursor_]);
        cursor_ = (cursor_ + 1) % samples_.size();
    }
    return batch;
}

}  // namespace lshaped
