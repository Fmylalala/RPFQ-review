#include "json_utils.h"

#include <cctype>
#include <fstream>
#include <sstream>

namespace rpfq_vit {
namespace {

class Parser {
public:
    explicit Parser(const std::string& input) : text(input) {}

    JsonValue parse() {
        skipWs();
        JsonValue value = parseValue();
        skipWs();
        if (pos != text.size()) {
            throw std::runtime_error("Trailing content in JSON.");
        }
        return value;
    }

private:
    const std::string& text;
    size_t pos = 0;

    void skipWs() {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
    }

    char peek() const {
        if (pos >= text.size()) {
            throw std::runtime_error("Unexpected end of JSON.");
        }
        return text[pos];
    }

    char take() {
        char c = peek();
        ++pos;
        return c;
    }

    void expect(char expected) {
        char c = take();
        if (c != expected) {
            throw std::runtime_error("Unexpected JSON character.");
        }
    }

    bool consumeLiteral(const char* literal) {
        size_t i = 0;
        while (literal[i] != '\0') {
            if (pos + i >= text.size() || text[pos + i] != literal[i]) {
                return false;
            }
            ++i;
        }
        pos += i;
        return true;
    }

    JsonValue parseValue() {
        skipWs();
        char c = peek();
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') {
            JsonValue v;
            v.type = JsonValue::Type::String;
            v.string_value = parseString();
            return v;
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parseNumber();
        if (consumeLiteral("true")) {
            JsonValue v;
            v.type = JsonValue::Type::Bool;
            v.bool_value = true;
            return v;
        }
        if (consumeLiteral("false")) {
            JsonValue v;
            v.type = JsonValue::Type::Bool;
            v.bool_value = false;
            return v;
        }
        if (consumeLiteral("null")) {
            return JsonValue{};
        }
        throw std::runtime_error("Invalid JSON value.");
    }

    JsonValue parseObject() {
        JsonValue v;
        v.type = JsonValue::Type::Object;
        expect('{');
        skipWs();
        if (peek() == '}') {
            ++pos;
            return v;
        }
        while (true) {
            skipWs();
            std::string key = parseString();
            skipWs();
            expect(':');
            v.object_value.emplace(std::move(key), parseValue());
            skipWs();
            char c = take();
            if (c == '}') break;
            if (c != ',') throw std::runtime_error("Expected comma in JSON object.");
        }
        return v;
    }

    JsonValue parseArray() {
        JsonValue v;
        v.type = JsonValue::Type::Array;
        expect('[');
        skipWs();
        if (peek() == ']') {
            ++pos;
            return v;
        }
        while (true) {
            v.array_value.push_back(parseValue());
            skipWs();
            char c = take();
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("Expected comma in JSON array.");
        }
        return v;
    }

    std::string parseString() {
        expect('"');
        std::string out;
        while (true) {
            char c = take();
            if (c == '"') break;
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            char esc = take();
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u':
                    if (pos + 4 > text.size()) throw std::runtime_error("Invalid unicode escape.");
                    out.push_back('?');
                    pos += 4;
                    break;
                default:
                    throw std::runtime_error("Invalid JSON escape.");
            }
        }
        return out;
    }

    JsonValue parseNumber() {
        size_t start = pos;
        if (text[pos] == '-') ++pos;
        while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) ++pos;
        if (pos < text.size() && text[pos] == '.') {
            ++pos;
            while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) ++pos;
        }
        if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
            ++pos;
            if (pos < text.size() && (text[pos] == '+' || text[pos] == '-')) ++pos;
            while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) ++pos;
        }
        JsonValue v;
        v.type = JsonValue::Type::Number;
        v.number_value = std::stod(text.substr(start, pos - start));
        return v;
    }
};

} // namespace

const JsonValue& JsonValue::at(const std::string& key) const {
    const JsonValue* found = find(key);
    if (!found) {
        throw std::runtime_error("Missing JSON key: " + key);
    }
    return *found;
}

const JsonValue* JsonValue::find(const std::string& key) const {
    if (!isObject()) return nullptr;
    auto it = object_value.find(key);
    if (it == object_value.end()) return nullptr;
    return &it->second;
}

std::string JsonValue::asString(const std::string& fallback) const {
    return isString() ? string_value : fallback;
}

int JsonValue::asInt(int fallback) const {
    return isNumber() ? static_cast<int>(number_value) : fallback;
}

double JsonValue::asDouble(double fallback) const {
    return isNumber() ? number_value : fallback;
}

bool JsonValue::asBool(bool fallback) const {
    return isBool() ? bool_value : fallback;
}

JsonValue parseJson(const std::string& text) {
    return Parser(text).parse();
}

JsonValue parseJsonFile(const std::string& path) {
    return parseJson(readTextFile(path));
}

std::string readTextFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace rpfq_vit
