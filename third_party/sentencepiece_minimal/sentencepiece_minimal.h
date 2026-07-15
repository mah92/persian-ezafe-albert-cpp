// sentencepiece_minimal.h
// Minimal header-only-ready SentencePieceProcessor with Load + Encode only.
// Zero dependencies beyond C++17 standard library.
// Optimized for Persian text but works for any language.

#ifndef SENTENCEPIECE_MINIMAL_H_
#define SENTENCEPIECE_MINIMAL_H_

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sentencepiece_minimal {

// ============================================================
// Result types
// ============================================================
struct EncodeResult {
    std::vector<std::string> pieces;
    std::vector<int> ids;
};

// ============================================================
// Model types (mirrors sentencepiece ModelProto without protobuf)
// ============================================================
enum class PieceType : int {
    NORMAL = 1,
    UNKNOWN = 2,
    CONTROL = 3,
    USER_DEFINED = 4,
    UNUSED = 5,
    BYTE = 6,
};

enum class ModelType : int {
    UNIGRAM = 1,
    BPE = 2,
    WORD = 3,
    CHAR = 4,
};

struct SentencePiece {
    std::string piece;
    float score = 0.0f;
    PieceType type = PieceType::NORMAL;
};

// ============================================================
// Minimal DoubleArray (Darts-clone subset) for trie lookup
// Only implements what we need: set_array and commonPrefixSearch
// ============================================================
class DoubleArray {
public:
    DoubleArray() = default;

    // Takes ownership of data (must be 4-byte aligned array of 32-bit units).
    void set_array(const char* data, size_t num_units);

    // commonPrefixSearch: find all keys that are prefixes of `key`.
    // Returns number of matches found (capped at result_size).
    struct result_pair {
        int length;
        int value;
    };
    size_t commonPrefixSearch(const char* key, result_pair* results,
                              size_t result_size, size_t key_len) const;

    size_t unit_size() const { return sizeof(uint32_t); }

private:
    const uint32_t* array_ = nullptr;
    size_t num_units_ = 0;

    // Inline helpers (matching darts.h)
    bool has_leaf(uint32_t unit) const { return (unit >> 8) & 1; }
    uint32_t value(uint32_t unit) const { return unit & 0x7FFFFFFF; }
    int label(uint32_t unit) const { return unit & 0xFF; }
    int offset(uint32_t unit) const;
};

// ============================================================
// SentencePieceProcessor - minimal implementation
// ============================================================
class SentencePieceProcessor {
public:
    SentencePieceProcessor() = default;
    ~SentencePieceProcessor() = default;

    // Load model from file path. Returns true on success.
    bool Load(const std::string& filename);

    // Load model from serialized protobuf bytes.
    bool LoadFromSerialized(std::string_view data);

    // Encode: tokenize UTF-8 text into pieces + ids.
    // For BPE models, uses standard greedy BPE.
    // For unigram models, uses Viterbi decoding.
    EncodeResult Encode(std::string_view input) const;

    // Vocabulary access.
    int GetPieceSize() const { return static_cast<int>(pieces_.size()); }
    const SentencePiece& GetPiece(int id) const { return pieces_[id]; }
    int PieceToId(std::string_view piece) const;

    // Model info.
    ModelType model_type() const { return model_type_; }
    bool IsByteFallback() const { return byte_fallback_; }

    // Public for debugging: apply normalization to input.
    std::string NormalizeText(std::string_view input) const {
        std::string out;
        Normalize(input, &out);
        return out;
    }

    // Read a varint from buffer, returns bytes consumed.
    static size_t ReadVarint(const uint8_t* data, size_t len, uint64_t* value);
    static size_t ReadVarint32(const uint8_t* data, size_t len, uint32_t* value);
    static float ReadFloat(const uint8_t* data);
    static int32_t ReadInt32(const uint8_t* data);

private:
    // Protobuf wire format parser helpers.
    bool ParseModelProto(std::string_view data);
    bool ParseTrainerSpec(std::string_view data);
    bool ParseNormalizerSpec(std::string_view data);
    bool ParseSentencePiece(std::string_view data, SentencePiece& sp);

    // Normalization.
    void Normalize(std::string_view input, std::string* normalized) const;
    void NormalizePrefix(const char* data, size_t len,
                         const char** out_start, size_t* out_len,
                         size_t* consumed_input) const;
    int PrefixMatchUserDefined(const char* data, size_t len) const;

    // BPE encoding.
    EncodeResult EncodeBPE(std::string_view normalized) const;

    // Unigram Viterbi encoding.
    EncodeResult EncodeUnigram(std::string_view normalized) const;

    // Model data.
    ModelType model_type_ = ModelType::BPE;
    std::vector<SentencePiece> pieces_;
    std::unordered_map<std::string, int> piece_to_id_;
    int unk_id_ = 0;
    int bos_id_ = -1;
    int eos_id_ = -1;
    int pad_id_ = -1;

    // Trainer spec fields we care about.
    bool byte_fallback_ = false;
    bool treat_whitespace_as_suffix_ = false;
    int max_piece_len_ = 16;

    // Normalizer spec fields.
    bool add_dummy_prefix_ = true;
    bool remove_extra_whitespaces_ = true;
    bool escape_whitespaces_ = true;

    // DoubleArray for normalization (charsmap).
    DoubleArray norm_trie_;
    std::string norm_output_;  // null-delimited replacement strings
    std::string norm_trie_data_;     // backing storage for the trie data
    std::vector<uint32_t> norm_trie_swap_buffer_;  // backing for swapped trie

    // User-defined symbols (linear scan, simpler than trie).
    std::vector<std::string> user_defined_symbols_;

    // Subword regex pattern check (simplified).
    bool split_by_unicode_script_ = false;
};

}  // namespace sentencepiece_minimal

#endif  // SENTENCEPIECE_MINIMAL_H_