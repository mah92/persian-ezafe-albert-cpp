#include "ezafe_detector.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

EzafeDetector::EzafeDetector(const std::string& onnx_path,
                             const std::string& spiece_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ezafe")
{
    // Load SentencePiece tokenizer
    if (!tokenizer_.Load(spiece_path))
        throw std::runtime_error("Failed to load SentencePiece model: " + spiece_path);

    // Load ONNX model (same pattern as hush_enhance_onnx.cpp)
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_options);
}

std::vector<EzafeDetector::Result> EzafeDetector::predict(
    const std::vector<std::string>& words)
{
    // 1) Tokenize each word individually via SentencePiece, collecting sub-token IDs.
    //    Track word boundaries (same logic as HuggingFace word_ids()).
    std::vector<int64_t> input_ids;
    std::vector<int> word_ids;   // maps each token position → original word index
    input_ids.push_back(kClsId);
    word_ids.push_back(-1);      // [CLS] → no word

    for (size_t wi = 0; wi < words.size(); ++wi) {
        auto encode_result = tokenizer_.Encode(words[wi]);
        for (auto id : encode_result.ids) {
            input_ids.push_back(static_cast<int64_t>(id));
            word_ids.push_back(static_cast<int>(wi));
        }
    }

    input_ids.push_back(kSepId);
    word_ids.push_back(-1);      // [SEP] → no word

    int64_t seq_len = static_cast<int64_t>(input_ids.size());
    if (seq_len > kMaxLength)
        seq_len = kMaxLength;

    // 2) Build attention_mask and token_type_ids (all zeros)
    std::vector<int64_t> attention_mask(seq_len, 1);
    std::vector<int64_t> token_type_ids(seq_len, 0);

    // Resize if truncated
    input_ids.resize(seq_len);
    // Note: if truncation removed the [SEP], that's acceptable

    // Pad to kMaxLength
    while (static_cast<int64_t>(input_ids.size()) < kMaxLength) {
        input_ids.push_back(kPadId);
        attention_mask.push_back(0);
        token_type_ids.push_back(0);
    }

    // 3) Create ONNX tensors (same pattern as hush_enhance_onnx.cpp Ort::Value::CreateTensor)
    std::vector<int64_t> shape = {1, kMaxLength};

    Ort::Value input_ort = Ort::Value::CreateTensor<int64_t>(
        mem_info_, input_ids.data(), input_ids.size(), shape.data(), shape.size());
    Ort::Value mask_ort = Ort::Value::CreateTensor<int64_t>(
        mem_info_, attention_mask.data(), attention_mask.size(), shape.data(), shape.size());
    Ort::Value type_ort = Ort::Value::CreateTensor<int64_t>(
        mem_info_, token_type_ids.data(), token_type_ids.size(), shape.data(), shape.size());

    // 4) Run inference
    const char* input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
    const char* output_names[] = {"logits"};

    std::array<Ort::Value, 3> inputs = {
        std::move(input_ort),
        std::move(mask_ort),
        std::move(type_ort)
    };

    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{},
        input_names, inputs.data(), 3,
        output_names, 1);

    // 5) Extract logits: shape [1, kMaxLength, 2]
    float* logits_data = outputs[0].GetTensorMutableData<float>();

    // 6) Map results back to original words (softmax + take first sub-token per word)
    std::vector<Result> results(words.size());
    for (size_t i = 0; i < words.size(); ++i)
        results[i].word = words[i];   // default

    std::vector<bool> word_seen(words.size(), false);

    for (int64_t t = 0; t < seq_len; ++t) {
        int wi = (t < static_cast<int64_t>(word_ids.size())) ? word_ids[t] : -1;
        if (wi < 0 || wi >= static_cast<int>(words.size()))
            continue;

        if (word_seen[wi])
            continue;   // use first sub-token only
        word_seen[wi] = true;

        // Softmax: only 2 classes, compute exp(x0) and exp(x1)
        float x0 = logits_data[t * 2 + 0];
        float x1 = logits_data[t * 2 + 1];
        float max_x = std::max(x0, x1);
        float e0 = std::exp(x0 - max_x);
        float e1 = std::exp(x1 - max_x);
        float sum = e0 + e1;
        float p_ezafe = e1 / sum;

        results[wi].needs_ezafe = (p_ezafe >= 0.5f);
        results[wi].confidence = p_ezafe;
    }

    return results;
}