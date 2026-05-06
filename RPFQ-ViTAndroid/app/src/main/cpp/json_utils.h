#pragma once

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace rpfq_vit {

class JsonValue {
public:
    enum class Type {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object
    };

    Type type = Type::Null;
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::map<std::string, JsonValue> object_value;

    bool isNull() const { return type == Type::Null; }
    bool isObject() const { return type == Type::Object; }
    bool isArray() const { return type == Type::Array; }
    bool isString() const { return type == Type::String; }
    bool isNumber() const { return type == Type::Number; }
    bool isBool() const { return type == Type::Bool; }

    const JsonValue& at(const std::string& key) const;
    const JsonValue* find(const std::string& key) const;
    std::string asString(const std::string& fallback = "") const;
    int asInt(int fallback = 0) const;
    double asDouble(double fallback = 0.0) const;
    bool asBool(bool fallback = false) const;
};

JsonValue parseJson(const std::string& text);
JsonValue parseJsonFile(const std::string& path);
std::string readTextFile(const std::string& path);

} // namespace rpfq_vit
