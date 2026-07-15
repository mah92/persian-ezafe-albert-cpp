#include "sentencepiece_minimal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <utility>

namespace sentencepiece_minimal {

// ============================================================
// UTF-8 helpers
// ============================================================
namespace {

// Length of a single UTF-8 character from first byte.
inline size_t OneCharLen(const char* src) {
    return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

}  // namespace

// ============================================================
// DoubleArray implementation (minimal Darts-clone subset)
// ============================================================

void DoubleArray::set_array(const char* data, size_t num_units) {
    array_ = reinterpret_cast<const uint32_t*>(data);
    num_units_ = num_units;
}

int DoubleArray::offset(uint32_t unit) const {
    return static_cast<int>((unit >> 10) << ((unit >> 9) & 1));
}

size_t DoubleArray::commonPrefixSearch(const char* key, result_pair* results,
                                        size_t result_size, size_t key_len) const {
    if (array_ == nullptr || num_units_ == 0) return 0;
    if (key_len == 0) return 0;

    size_t num_results = 0;
    size_t node_pos = 0;
    uint32_t unit = array_[node_pos];
    node_pos ^= offset(unit);

    for (size_t i = 0; i < key_len; ++i) {
        const unsigned char c = static_cast<unsigned char>(key[i]);
        node_pos ^= c;
        unit = array_[node_pos];
        if (label(unit) != c) {
            return num_results;
        }
        node_pos ^= offset(unit);
        if (has_leaf(unit)) {
            if (num_results < result_size) {
                results[num_results].length = static_cast<int>(i + 1);
                results[num_results].value = value(unit);
            }
            ++num_results;
        }
    }
    return num_results;
}

// ============================================================
// Protobuf wire-format parser
// ============================================================

size_t SentencePieceProcessor::ReadVarint(const uint8_t* data, size_t len, uint64_t* value) {
    *value = 0;
    size_t shift = 0;
    size_t i = 0;
    while (i < len) {
        uint8_t byte = data[i++];
        *value |= static_cast<uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) break;
        shift += 7;
    }
    return i;
}

size_t SentencePieceProcessor::ReadVarint32(const uint8_t* data, size_t len, uint32_t* value) {
    uint64_t v = 0;
    size_t consumed = ReadVarint(data, len, &v);
    *value = static_cast<uint32_t>(v);
    return consumed;
}

float SentencePieceProcessor::ReadFloat(const uint8_t* data) {
    float f;
    uint32_t bits = static_cast<uint32_t>(data[0]) |
                    (static_cast<uint32_t>(data[1]) << 8) |
                    (static_cast<uint32_t>(data[2]) << 16) |
                    (static_cast<uint32_t>(data[3]) << 24);
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Protobuf wire types
enum WireType {
    WIRE_VARINT = 0,
    WIRE_64BIT = 1,
    WIRE_LENGTH_DELIMITED = 2,
    WIRE_32BIT = 5,
};

// Read a protobuf field tag. Returns (field_number, wire_type).
static std::pair<uint32_t, uint8_t> ReadTag(const uint8_t* data, size_t len, size_t* consumed) {
    uint64_t tag = 0;
    *consumed = SentencePieceProcessor::ReadVarint(data, len, &tag);
    return {static_cast<uint32_t>(tag >> 3), static_cast<uint8_t>(tag & 0x7)};
}

bool SentencePieceProcessor::ParseSentencePiece(std::string_view data, SentencePiece& sp) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data());
    const uint8_t* end = p + data.size();
    sp = SentencePiece{};

    while (p < end) {
        size_t consumed = 0;
        auto [field, wire] = ReadTag(p, end - p, &consumed);
        p += consumed;
        if (p > end) break;

        switch (field) {
            case 1: {  // piece (string)
                if (wire != WIRE_LENGTH_DELIMITED) return false;
                uint32_t len = 0;
                size_t c = ReadVarint32(p, end - p, &len);
                p += c;
                if (p + len > end) return false;
                sp.piece.assign(reinterpret_cast<const char*>(p), len);
                p += len;
                break;
            }
            case 2: {  // score (float)
                if (wire != WIRE_32BIT) return false;
                if (p + 4 > end) return false;
                sp.score = ReadFloat(p);
                p += 4;
                break;
            }
            case 3: {  // type (enum/int)
                if (wire != WIRE_VARINT) return false;
                uint32_t v = 0;
                size_t c = ReadVarint32(p, end - p, &v);
                p += c;
                sp.type = static_cast<PieceType>(v);
                break;
            }
            default:
                // Skip unknown field
                if (wire == WIRE_VARINT) {
                    uint64_t dummy;
                    p += ReadVarint(p, end - p, &dummy);
                } else if (wire == WIRE_64BIT) {
                    p += 8;
                } else if (wire == WIRE_32BIT) {
                    p += 4;
                } else if (wire == WIRE_LENGTH_DELIMITED) {
                    uint32_t len = 0;
                    size_t c = ReadVarint32(p, end - p, &len);
                    p += c + len;
                } else {
                    return false;
                }
                break;
        }
    }
    return !sp.piece.empty();
}

bool SentencePieceProcessor::ParseTrainerSpec(std::string_view data) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data());
    const uint8_t* end = p + data.size();

    while (p < end) {
        size_t consumed = 0;
        auto [field, wire] = ReadTag(p, end - p, &consumed);
        p += consumed;
        if (p > end) break;

        switch (field) {
            case 3: {  // model_type
                if (wire != WIRE_VARINT) break;
                uint32_t v = 0;
                size_t c = ReadVarint32(p, end - p, &v);
                p += c;
                model_type_ = static_cast<ModelType>(v);
                break;
            }
            case 20: {  // max_sentencepiece_length
                if (wire != WIRE_VARINT) break;
                uint32_t v = 0;
                size_t c = ReadVarint32(p, end - p, &v);
                p += c;
                if (v > 0 && v < 512) max_piece_len_ = static_cast<int>(v);
                break;
            }
            case 24: {  // treat_whitespace_as_suffix
                if (wire != WIRE_VARINT) break;
                uint32_t v = 0;
                size_t c = ReadVarint32(p, end - p, &v);
                p += c;
                treat_whitespace_as_suffix_ = (v != 0);
                break;
            }
            case 35: {  // byte_fallback
                if (wire != WIRE_VARINT) break;
                uint32_t v = 0;
                size_t c = ReadVarint32(p, end - p, &v);
                p += c;
                byte_fallback_ = (v != 0);
                break;
            }
            case 40: {  // unk_id
                if (wire != WIRE_VARINT) break;
                uint32_t v = 0;
                size_t c = ReadVarint32(p, end - p, &v);
                p += c;
                unk_id_ = static_cast<int>(v);
                break;
            }
            case 41: {  // bos_id
                if (wire != WIRE_VARINT) break;
                uint32_t v = 0;
                size_t c = ReadVarint32(p, end - p, &v);
                p += c;
                bos_id_ = static_cast<int>(v);
                break;
            }
            case 42: {  // eos_id
                if (wire != WIRE_VARINT) break;
                uint32_t v = 0;
                size_t c = ReadVarint32(p, end - p, &v);
                p += c;
                eos_id_ = static_cast<int>(v);
                break;
            }
            case 43: {  // pad_id
                if (wire != WIRE_VARINT) break;
                uint32_t v = 0;
                size_t c = ReadVarint32(p, end - p, &v);
                p += c;
                pad_id_ = static_cast<int>(v);
                break;
            }
            default:
                if (wire == WIRE_VARINT) {
                    uint64_t dummy;
                    p += ReadVarint(p, end - p, &dummy);
                } else if (wire == WIRE_64BIT) {
                    p += 8;
                } else if (wire == WIRE_32BIT) {
                    p += 4;
                } else if (wire == WIRE_LENGTH_DELIMITED) {
                    uint32_t len = 0;
                    size_t c = ReadVarint32(p, end - p, &len);
                    p += c + len;
                } else {
                    break;
                }
                break;
        }
    }
    return true;
}

bool SentencePieceProcessor::ParseNormalizerSpec(std::string_view data) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data());
    const uint8_t* end = p + data.size();

    while (p < end) {
        size_t consumed = 0;
        auto [field, wire] = ReadTag(p, end - p, &consumed);
        p += consumed;
        if (p > end) break;

        switch (field) {
            case 2: {  // precompiled_charsmap (bytes)
                if (wire != WIRE_LENGTH_DELIMITED) break;
                uint32_t len = 0;
                size_t c = ReadVarint32(p, end - p, &len);
                p += c;
                if (p + len > end) break;

                if (len > sizeof(uint32_t)) {
                    // Try native (LE) endian first, then BE fallback.
                    uint32_t trie_blob_size;
                    std::memcpy(&trie_blob_size, p, sizeof(uint32_t));

                    bool valid = (trie_blob_size >= 1024 && (trie_blob_size & 0x3FF) == 0 &&
                                  trie_blob_size + sizeof(uint32_t) <= len);
                    bool needs_swap = false;

                    if (!valid) {
                        // Try big-endian
                        trie_blob_size = (static_cast<uint32_t>(p[0]) << 24) |
                                         (static_cast<uint32_t>(p[1]) << 16) |
                                         (static_cast<uint32_t>(p[2]) << 8) |
                                         static_cast<uint32_t>(p[3]);
                        if (trie_blob_size >= 1024 && (trie_blob_size & 0x3FF) == 0 &&
                            trie_blob_size + sizeof(uint32_t) <= len) {
                            needs_swap = true;
                            valid = true;
                        }
                    }

                    if (valid) {
                        size_t num_units = trie_blob_size / 4;
                        const uint8_t* trie_data = p + sizeof(uint32_t);

                        // Always copy to aligned vector<uint32_t>.
                        norm_trie_swap_buffer_.resize(num_units);
                        for (size_t i = 0; i < num_units; ++i) {
                            const uint8_t* src = trie_data + i * 4;
                            if (needs_swap) {
                                norm_trie_swap_buffer_[i] =
                                    (static_cast<uint32_t>(src[0]) << 24) |
                                    (static_cast<uint32_t>(src[1]) << 16) |
                                    (static_cast<uint32_t>(src[2]) << 8) |
                                    static_cast<uint32_t>(src[3]);
                            } else {
                                std::memcpy(&norm_trie_swap_buffer_[i], src, 4);
                            }
                        }
                        norm_trie_.set_array(
                            reinterpret_cast<const char*>(norm_trie_swap_buffer_.data()),
                            num_units);

                        const char* norm_start =
                            reinterpret_cast<const char*>(trie_data + trie_blob_size);
                        size_t norm_len = len - sizeof(uint32_t) - trie_blob_size;
                        norm_output_.assign(norm_start, norm_len);
                    }
                }
                p += len;
                break;
            }
            case 3: {  // add_dummy_prefix
                if (wire != WIRE_VARINT) break;
                uint32_t v = 0;
                size_t c = ReadVarint32(p, end - p, &v);
                p += c;
                add_dummy_prefix_ = (v != 0);
                break;
            }
            case 4: {  // remove_extra_whitespaces
                if (wire != WIRE_VARINT) break;
                uint32_t v = 0;
                size_t c = ReadVarint32(p, end - p, &v);
                p += c;
                remove_extra_whitespaces_ = (v != 0);
                break;
            }
            case 5: {  // escape_whitespaces
                if (wire != WIRE_VARINT) break;
                uint32_t v = 0;
                size_t c = ReadVarint32(p, end - p, &v);
                p += c;
                escape_whitespaces_ = (v != 0);
                break;
            }
            default:
                if (wire == WIRE_VARINT) {
                    uint64_t dummy;
                    p += ReadVarint(p, end - p, &dummy);
                } else if (wire == WIRE_64BIT) {
                    p += 8;
                } else if (wire == WIRE_32BIT) {
                    p += 4;
                } else if (wire == WIRE_LENGTH_DELIMITED) {
                    uint32_t l = 0;
                    size_t c2 = ReadVarint32(p, end - p, &l);
                    p += c2 + l;
                } else {
                    break;
                }
                break;
        }
    }
    return true;
}

bool SentencePieceProcessor::ParseModelProto(std::string_view data) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data());
    const uint8_t* end = p + data.size();

    std::vector<SentencePiece> pieces;
    std::string_view trainer_spec_data;
    std::string_view normalizer_spec_data;
    std::vector<std::string> user_defined_symbols;

    while (p < end) {
        size_t consumed = 0;
        auto [field, wire] = ReadTag(p, end - p, &consumed);
        p += consumed;
        if (p > end) break;

        switch (field) {
            case 1: {  // pieces (repeated SentencePiece)
                if (wire != WIRE_LENGTH_DELIMITED) break;
                uint32_t len = 0;
                size_t c = ReadVarint32(p, end - p, &len);
                p += c;
                if (p + len > end) break;
                SentencePiece sp;
                if (ParseSentencePiece(std::string_view(reinterpret_cast<const char*>(p), len), sp)) {
                    pieces.push_back(std::move(sp));
                }
                p += len;
                break;
            }
            case 2: {  // trainer_spec
                if (wire != WIRE_LENGTH_DELIMITED) break;
                uint32_t len = 0;
                size_t c = ReadVarint32(p, end - p, &len);
                p += c;
                if (p + len > end) break;
                trainer_spec_data = std::string_view(reinterpret_cast<const char*>(p), len);
                p += len;
                break;
            }
            case 3: {  // normalizer_spec
                if (wire != WIRE_LENGTH_DELIMITED) break;
                uint32_t len = 0;
                size_t c = ReadVarint32(p, end - p, &len);
                p += c;
                if (p + len > end) break;
                normalizer_spec_data = std::string_view(reinterpret_cast<const char*>(p), len);
                p += len;
                break;
            }
            default:
                if (wire == WIRE_VARINT) {
                    uint64_t dummy;
                    p += ReadVarint(p, end - p, &dummy);
                } else if (wire == WIRE_64BIT) {
                    p += 8;
                } else if (wire == WIRE_32BIT) {
                    p += 4;
                } else if (wire == WIRE_LENGTH_DELIMITED) {
                    uint32_t l = 0;
                    size_t c2 = ReadVarint32(p, end - p, &l);
                    p += c2 + l;
                } else {
                    break;
                }
                break;
        }
    }

    if (pieces.empty()) return false;

    if (!trainer_spec_data.empty()) {
        ParseTrainerSpec(trainer_spec_data);
    }
    if (!normalizer_spec_data.empty()) {
        ParseNormalizerSpec(normalizer_spec_data);
    }

    pieces_ = std::move(pieces);
    for (int i = 0; i < static_cast<int>(pieces_.size()); ++i) {
        piece_to_id_[pieces_[i].piece] = i;
        if (pieces_[i].type == PieceType::USER_DEFINED) {
            user_defined_symbols.push_back(pieces_[i].piece);
        }
        if (pieces_[i].type == PieceType::UNKNOWN && unk_id_ == 0) {
            unk_id_ = i;
        }
    }

    if (!user_defined_symbols.empty()) {
        user_defined_symbols_ = std::move(user_defined_symbols);
    }

    return true;
}

bool SentencePieceProcessor::Load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string buffer(static_cast<size_t>(size), '\0');
    if (!file.read(buffer.data(), size)) return false;

    return LoadFromSerialized(buffer);
}

bool SentencePieceProcessor::LoadFromSerialized(std::string_view data) {
    return ParseModelProto(data);
}

// ============================================================
// Normalization
// ============================================================

int SentencePieceProcessor::PrefixMatchUserDefined(const char* data, size_t len) const {
    if (user_defined_symbols_.empty()) return 0;
    int best_len = 0;
    for (const auto& sym : user_defined_symbols_) {
        if (sym.size() <= len &&
            std::memcmp(data, sym.data(), sym.size()) == 0 &&
            static_cast<int>(sym.size()) > best_len) {
            best_len = static_cast<int>(sym.size());
        }
    }
    return best_len;
}

void SentencePieceProcessor::NormalizePrefix(const char* data, size_t len,
                                              const char** out_start, size_t* out_len,
                                              size_t* consumed_input) const {
    // 1. Check user-defined symbols first (they are protected)
    int user_len = PrefixMatchUserDefined(data, len);
    if (user_len > 0) {
        *out_start = data;
        *out_len = static_cast<size_t>(user_len);
        *consumed_input = static_cast<size_t>(user_len);
        return;
    }

    // 2. Check normalization trie (charsmap) for NFKC normalization.
    //   DISABLED: trie byte-order handling is platform-dependent
    //   and produces incorrect replacements for Arabic diacritics.
    //   Character-by-character pass-through is correct for our Persian model.
    if (false && !norm_output_.empty()) {
        constexpr int kMaxResults = 32;
        DoubleArray::result_pair results[kMaxResults];
        size_t n = norm_trie_.commonPrefixSearch(data, results, kMaxResults, len);
        if (n > kMaxResults) n = kMaxResults;

        int best_len = 0;
        int best_value = 0;
        for (size_t i = 0; i < n; ++i) {
            if (results[i].length > best_len) {
                best_len = results[i].length;
                best_value = results[i].value;
            }
        }

        if (best_len > 0 && best_value >= 0 &&
            static_cast<size_t>(best_value) < norm_output_.size()) {
            const char* replacement = norm_output_.data() + best_value;
            size_t rep_len = 0;
            while (best_value + rep_len < norm_output_.size() &&
                   replacement[rep_len] != '\0') {
                ++rep_len;
            }
            // If replacement is empty, fall through to default
            if (rep_len > 0) {
                *out_start = replacement;
                *out_len = rep_len;
                *consumed_input = static_cast<size_t>(best_len);
                return;
            }
        }
    }

    // 3. Default: one UTF-8 character
    size_t clen = OneCharLen(data);
    if (clen > len) clen = len;
    *out_start = data;
    *out_len = clen;
    *consumed_input = clen;
}

void SentencePieceProcessor::Normalize(std::string_view input,
                                        std::string* normalized) const {
    normalized->clear();
    if (input.empty()) return;

    normalized->reserve(input.size() * 1.5);

    const char* kSpaceSymbol = escape_whitespaces_ ? "\xe2\x96\x81" : " ";  // ▁ or space

    const char* p = input.data();
    const char* end = p + input.size();

    // Remove heading whitespace
    if (remove_extra_whitespaces_) {
        while (p < end) {
            const char* out;
            size_t out_len, consumed;
            NormalizePrefix(p, end - p, &out, &out_len, &consumed);
            std::string_view sv(out, out_len);
            if (sv != " ") break;
            p += consumed;
        }
    }

    if (p >= end) return;

    // Add dummy prefix (default = true)
    if (!treat_whitespace_as_suffix_ && add_dummy_prefix_) {
        normalized->append(kSpaceSymbol);
    }

    bool is_prev_space = remove_extra_whitespaces_;
    while (p < end) {
        const char* out;
        size_t out_len, consumed_input;
        NormalizePrefix(p, end - p, &out, &out_len, &consumed_input);

        std::string_view sp(out, out_len);

        // Remove heading spaces if previous was space
        while (is_prev_space && !sp.empty() && sp[0] == ' ') {
            sp.remove_prefix(1);
        }

        if (!sp.empty()) {
            for (size_t i = 0; i < sp.size(); ++i) {
                if (sp[i] == ' ') {
                    normalized->append(kSpaceSymbol);
                } else {
                    *normalized += sp[i];
                }
            }
            is_prev_space = (sp.back() == ' ');
        }

        p += consumed_input;
        if (!remove_extra_whitespaces_) {
            is_prev_space = false;
        }
    }

    // Remove trailing whitespace
    if (remove_extra_whitespaces_) {
        size_t klen = std::strlen(kSpaceSymbol);
        while (normalized->size() >= klen &&
               normalized->compare(normalized->size() - klen, klen, kSpaceSymbol) == 0) {
            normalized->resize(normalized->size() - klen);
        }
    }

    // Add dummy suffix if needed
    if (treat_whitespace_as_suffix_ && add_dummy_prefix_) {
        normalized->append(kSpaceSymbol);
    }
}

// ============================================================
// BPE Encode
// ============================================================

// Symbol in the BPE linked list. Uses indices into a persistent string pool
// (std::deque) to avoid dangling string_view references.
struct BPESymbol {
    int prev;
    int next;
    bool freeze;
    size_t piece_idx;   // index into merged_pool
    size_t piece_len;   // length of this piece
};

// Pair of symbols considered for merging
struct BPEPair {
    float score;
    int left;
    int right;
    int size;
};

struct BPEPairComparator {
    bool operator()(const BPEPair& a, const BPEPair& b) const {
        if (a.score != b.score) return a.score < b.score;
        return a.left > b.left;
    }
};

EncodeResult SentencePieceProcessor::EncodeBPE(std::string_view normalized) const {
    EncodeResult result;
    if (normalized.empty()) return result;

    // Persistent string pool (deque: no reallocation/invalidation on push_back)
    std::deque<std::string> pool;

    // Build initial symbols
    std::vector<BPESymbol> symbols;
    symbols.reserve(normalized.size());

    const char* p = normalized.data();
    const char* end = p + normalized.size();
    int index = 0;
    while (p < end) {
        BPESymbol s;
        int user_len = PrefixMatchUserDefined(p, end - p);
        s.freeze = (user_len > 0);
        if (s.freeze) {
            pool.push_back(std::string(p, user_len));
            s.piece_len = user_len;
            p += user_len;
        } else {
            size_t clen = OneCharLen(p);
            if (clen > static_cast<size_t>(end - p)) clen = end - p;
            pool.push_back(std::string(p, clen));
            s.piece_len = clen;
            p += clen;
        }
        s.piece_idx = pool.size() - 1;
        s.prev = (index == 0) ? -1 : index - 1;
        s.next = (p >= end) ? -1 : index + 1;
        ++index;
        symbols.push_back(s);
    }

    if (symbols.empty()) return result;

    // Helper to get piece string from pool
    auto piece_str = [&](const BPESymbol& s) -> std::string_view {
        return pool[s.piece_idx];
    };

    // Reverse merge map: merged string -> pair of original strings
    std::unordered_map<std::string, std::pair<std::string, std::string>> rev_merge;

    // Build initial agenda of adjacent pairs
    std::priority_queue<BPEPair, std::vector<BPEPair>, BPEPairComparator> agenda;

    for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i) {
        BPESymbol& left = symbols[i];
        BPESymbol& right = symbols[i + 1];
        if (left.freeze || right.freeze) continue;

        std::string_view l = piece_str(left);
        std::string_view r = piece_str(right);
        std::string merged(l);
        merged.append(r.data(), r.size());

        auto it = piece_to_id_.find(merged);
        if (it == piece_to_id_.end()) continue;
        int id = it->second;
        if (id >= static_cast<int>(pieces_.size())) continue;

        if (pieces_[id].type == PieceType::CONTROL ||
            pieces_[id].type == PieceType::BYTE ||
            (pieces_[id].type == PieceType::UNKNOWN && byte_fallback_)) {
            continue;
        }

        BPEPair pair;
        pair.score = pieces_[id].score;
        pair.left = i;
        pair.right = i + 1;
        pair.size = static_cast<int>(merged.size());
        agenda.push(pair);

        if (pieces_[id].type == PieceType::UNUSED) {
            rev_merge[merged] = {std::string(l), std::string(r)};
        }
    }

    // Greedy BPE merge loop
    while (!agenda.empty()) {
        const auto& top_ref = agenda.top();
        // Check if this pair is still valid
        if (top_ref.left >= static_cast<int>(symbols.size()) ||
            top_ref.right >= static_cast<int>(symbols.size())) {
            agenda.pop();
            continue;
        }
        BPESymbol& left_sym = symbols[top_ref.left];
        BPESymbol& right_sym = symbols[top_ref.right];
        std::string_view l = piece_str(left_sym);
        std::string_view r = piece_str(right_sym);
        if (l.empty() || r.empty() ||
            static_cast<int>(l.size() + r.size()) != top_ref.size) {
            agenda.pop();
            continue;
        }

        // Copy before pop — the reference becomes invalid after pop()
        BPEPair top = agenda.top();
        agenda.pop();

        // Merge: create new pooled string
        std::string merged(l);
        merged.append(r.data(), r.size());
        pool.push_back(std::move(merged));
        size_t new_idx = pool.size() - 1;

        left_sym.piece_idx = new_idx;
        left_sym.piece_len = l.size() + r.size();
        left_sym.next = right_sym.next;
        if (right_sym.next >= 0) {
            symbols[right_sym.next].prev = top.left;
        }
        // Mark right as consumed (zero length)
        right_sym.piece_len = 0;

        // Add new pairs involving the merged symbol
        auto add_pair = [&](int l_idx, int r_idx) {
            if (l_idx == -1 || r_idx == -1) return;
            BPESymbol& sl = symbols[l_idx];
            BPESymbol& sr = symbols[r_idx];
            if (sl.freeze || sr.freeze || sl.piece_len == 0 || sr.piece_len == 0) return;

            std::string_view slv = piece_str(sl);
            std::string_view srv = piece_str(sr);
            std::string m(slv);
            m.append(srv.data(), srv.size());

            auto it = piece_to_id_.find(m);
            if (it == piece_to_id_.end()) return;
            int id = it->second;
            if (id >= static_cast<int>(pieces_.size())) return;
            if (pieces_[id].type == PieceType::CONTROL ||
                pieces_[id].type == PieceType::BYTE ||
                (pieces_[id].type == PieceType::UNKNOWN && byte_fallback_)) {
                return;
            }

            BPEPair np;
            np.score = pieces_[id].score;
            np.left = l_idx;
            np.right = r_idx;
            np.size = static_cast<int>(m.size());
            agenda.push(np);

            if (pieces_[id].type == PieceType::UNUSED) {
                rev_merge[m] = {std::string(slv), std::string(srv)};
            }
        };

        add_pair(left_sym.prev, top.left);
        add_pair(top.left, left_sym.next);
    }

    // Resegment: expand UNUSED (intermediate) pieces back to subwords
    std::function<void(std::string_view, int)> resegment =
        [&](std::string_view w, int depth) {
            if (depth > 100) {
                std::string ps(w);
                auto it = piece_to_id_.find(ps);
                int id = (it != piece_to_id_.end()) ? it->second : unk_id_;
                result.pieces.push_back(std::move(ps));
                result.ids.push_back(id);
                return;
            }
            std::string ps(w);
            auto it = piece_to_id_.find(ps);
            int id = (it != piece_to_id_.end()) ? it->second : unk_id_;
            if (it == piece_to_id_.end() ||
                id >= static_cast<int>(pieces_.size()) ||
                pieces_[id].type != PieceType::UNUSED) {
                result.pieces.push_back(std::move(ps));
                result.ids.push_back(id);
                return;
            }
            auto rit = rev_merge.find(ps);
            if (rit == rev_merge.end()) {
                result.pieces.push_back(std::move(ps));
                result.ids.push_back(id);
                return;
            }
            resegment(rit->second.first, depth + 1);
            resegment(rit->second.second, depth + 1);
        };

    // Walk linked list and emit pieces
    for (int i = 0; i < static_cast<int>(symbols.size()); ++i) {
        if (symbols[i].piece_len == 0) continue;
        for (int idx = i; idx != -1; idx = symbols[idx].next) {
            if (idx >= 0 && idx < static_cast<int>(symbols.size()) &&
                symbols[idx].piece_len > 0) {
                resegment(piece_str(symbols[idx]), 0);
            }
        }
        break;
    }

    return result;
}

// ============================================================
// Unigram (Viterbi) Encode
// ============================================================

EncodeResult SentencePieceProcessor::EncodeUnigram(std::string_view normalized) const {
    EncodeResult result;
    if (normalized.empty()) return result;

    size_t n = normalized.size();
    std::vector<float> dp(n + 1, -std::numeric_limits<float>::infinity());
    std::vector<int> prev_piece(n + 1, -1);
    dp[0] = 0.0f;

    const int kMaxPieceLen = max_piece_len_ > 0 ? max_piece_len_ : 16;

    // kMaxPieceLen is in CHARACTERS, not bytes. Convert to byte limit.
    // We multiply by 4 for worst-case UTF-8 and clamp to normalized.size().
    size_t byte_max = std::min<size_t>(n, static_cast<size_t>(kMaxPieceLen) * 4);

    // Compute the size of a single UTF-8 char at position p, clamped to end.
    auto char_len_at = [&](size_t p) -> size_t {
        if (p >= n) return 0;
        size_t clen = OneCharLen(normalized.data() + p);
        if (clen > n - p) clen = n - p;
        return clen;
    };

    for (size_t pos = 0; pos < n; ++pos) {
        if (dp[pos] <= -1e30f) continue;
        size_t max_len = std::min<size_t>(n - pos, byte_max);
        for (size_t len = 1; len <= max_len; ++len) {
            // Ensure len ends at a valid UTF-8 character boundary
            size_t walked = 0;
            while (walked < len) {
                size_t step = char_len_at(pos + walked);
                if (walked + step > len) { walked = 0; break; }
                walked += step;
            }
            if (walked != len) continue;

            std::string piece(normalized.substr(pos, len));
            auto it = piece_to_id_.find(piece);
            int id;
            if (it != piece_to_id_.end()) {
                id = it->second;
                if (id >= static_cast<int>(pieces_.size())) continue;
                if (pieces_[id].type == PieceType::CONTROL ||
                    (pieces_[id].type == PieceType::BYTE && !byte_fallback_))
                    continue;
            } else {
                // Single UTF-8 character not in vocab: use UNK.
                // Multi-char substrings not in vocab: skip.
                if (char_len_at(pos) == len) {
                    id = unk_id_;
                } else {
                    continue;
                }
            }

            float new_prob = dp[pos] + pieces_[id].score;
            if (new_prob > dp[pos + len]) {
                dp[pos + len] = new_prob;
                prev_piece[pos + len] = static_cast<int>(pos);
            }
        }
    }

    // Backtrack
    if (dp[n] <= -1e30f) {
        // Fallback: character-level encoding
        size_t pos = 0;
        while (pos < n) {
            size_t clen = OneCharLen(normalized.data() + pos);
            if (clen > n - pos) clen = n - pos;
            std::string piece(normalized.substr(pos, clen));
            auto it = piece_to_id_.find(piece);
            int id = (it != piece_to_id_.end()) ? it->second : unk_id_;
            result.pieces.push_back(std::move(piece));
            result.ids.push_back(id);
            pos += clen;
        }
    } else {
        std::vector<int> positions;
        int pos = static_cast<int>(n);
        while (pos > 0) {
            positions.push_back(pos);
            pos = prev_piece[pos];
        }
        positions.push_back(0);
        std::reverse(positions.begin(), positions.end());

        for (size_t i = 0; i + 1 < positions.size(); ++i) {
            int start = positions[i];
            int end = positions[i + 1];
            std::string piece(normalized.substr(start, end - start));
            auto it = piece_to_id_.find(piece);
            int id = (it != piece_to_id_.end()) ? it->second : unk_id_;
            result.pieces.push_back(std::move(piece));
            result.ids.push_back(id);
        }
    }

    return result;
}

// ============================================================
// Encode (main entry point)
// ============================================================

EncodeResult SentencePieceProcessor::Encode(std::string_view input) const {
    if (pieces_.empty()) return {};

    std::string normalized;
    Normalize(input, &normalized);

    if (normalized.empty()) return {};

    switch (model_type_) {
        case ModelType::BPE:
            return EncodeBPE(normalized);
        case ModelType::UNIGRAM:
            return EncodeUnigram(normalized);
        case ModelType::CHAR:
            {
                EncodeResult result;
                size_t pos = 0;
                while (pos < normalized.size()) {
                    size_t clen = OneCharLen(normalized.data() + pos);
                    if (clen > normalized.size() - pos) clen = normalized.size() - pos;
                    std::string piece(normalized.substr(pos, clen));
                    auto it = piece_to_id_.find(piece);
                    int id = (it != piece_to_id_.end()) ? it->second : unk_id_;
                    result.pieces.push_back(std::move(piece));
                    result.ids.push_back(id);
                    pos += clen;
                }
                return result;
            }
        case ModelType::WORD:
            {
                EncodeResult result;
                size_t pos = 0;
                std::string current;
                while (pos < normalized.size()) {
                    size_t clen = OneCharLen(normalized.data() + pos);
                    if (clen > normalized.size() - pos) clen = normalized.size() - pos;
                    std::string_view ch(normalized.data() + pos, clen);
                    if (ch == "\xe2\x96\x81") {
                        if (!current.empty()) {
                            auto it = piece_to_id_.find(current);
                            int id = (it != piece_to_id_.end()) ? it->second : unk_id_;
                            result.pieces.push_back(std::move(current));
                            result.ids.push_back(id);
                            current.clear();
                        }
                    } else {
                        current.append(ch.data(), ch.size());
                    }
                    pos += clen;
                }
                if (!current.empty()) {
                    auto it = piece_to_id_.find(current);
                    int id = (it != piece_to_id_.end()) ? it->second : unk_id_;
                    result.pieces.push_back(std::move(current));
                    result.ids.push_back(id);
                }
                return result;
            }
    }
    return {};
}

int SentencePieceProcessor::PieceToId(std::string_view piece) const {
    auto it = piece_to_id_.find(std::string(piece));
    if (it != piece_to_id_.end()) return it->second;
    return unk_id_;
}

}  // namespace sentencepiece_minimal