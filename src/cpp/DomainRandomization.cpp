#include "DomainRandomization.h"
#include <tinyxml2.h>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <vector>
#include <set>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>

namespace fs = std::filesystem;
using tinyxml2::XMLElement;
namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("Domain randomization: " + message);
}
std::vector<double> numbers(const YAML::Node& node, size_t size) {
    std::vector<double> result;
    if (size == 1 && node.IsScalar()) result.push_back(node.as<double>());
    else {
        require(node.IsSequence() && node.size() == size, "expected vector of length " + std::to_string(size));
        for (const auto& item : node) result.push_back(item.as<double>());
    }
    for (double value : result) require(std::isfinite(value), "values must be finite");
    return result;
}
std::vector<double> attributeNumbers(XMLElement* element, const char* attr, size_t size) {
    require(element && element->Attribute(attr), std::string("missing XML attribute ") + attr);
    std::istringstream stream(element->Attribute(attr));
    std::vector<double> values(size);
    for (double& value : values) require(bool(stream >> value) && std::isfinite(value), "target attributes must be numeric literals");
    std::string extra;
    require(!(stream >> extra), "wrong number of XML attribute values");
    return values;
}
std::string format(const std::vector<double>& values) {
    std::ostringstream out; out << std::setprecision(17);
    for (size_t i = 0; i < values.size(); ++i) { if (i) out << ' '; out << values[i]; }
    return out.str();
}
void visit(XMLElement* node, const std::function<void(XMLElement*)>& fn) {
    if (!node) return;
    fn(node);
    for (auto* child = node->FirstChildElement(); child; child = child->NextSiblingElement()) visit(child, fn);
}
}

YAML::Node GenerateRandomizedScenario(const std::string& scene, const std::string& resources,
                                     const std::string& outputDirectory, const YAML::Node& targets, bool writeScenes) {
    require(targets.IsSequence() && (!writeScenes || targets.size() > 0), "targets must be a list (nonempty for scene generation)");
    const fs::path resourceRoot = fs::canonical(resources);
    fs::create_directories(outputDirectory);
    const fs::path out = fs::canonical(outputDirectory);
    auto resolve = [&](const std::string& path) {
        fs::path p(path);
        return fs::canonical(p.is_absolute() ? p : resourceRoot / p);
    };
    // Load and validate everything before writing generated XML.
    std::map<fs::path, std::unique_ptr<tinyxml2::XMLDocument>> docs;
    std::map<fs::path, fs::path> outputs;
    std::set<fs::path> loading;
    std::function<void(const fs::path&)> load = [&](const fs::path& path) {
        require(!loading.count(path), "cyclic scene includes");
        if (docs.count(path)) return;
        loading.insert(path);
        auto doc = std::make_unique<tinyxml2::XMLDocument>();
        require(doc->LoadFile(path.c_str()) == tinyxml2::XML_SUCCESS, "cannot parse " + path.string());
        require(doc->FirstChildElement("scenario"), "missing scenario root");
        auto* ptr = doc.get();
        outputs[path] = out / ("scene_" + std::to_string(docs.size()) + ".scn");
        docs[path] = std::move(doc);
        visit(ptr->RootElement(), [&](XMLElement* node) {
            if (std::string(node->Name()) == "include") {
                require(node->Attribute("file"), "include without file");
                load(resolve(node->Attribute("file")));
            }
        });
        loading.erase(path);
    };
    const fs::path main = resolve(scene);
    load(main);
    for (const auto& pair : outputs) {
        require(!docs.count(fs::weakly_canonical(pair.second)), "output would overwrite a source scene");
        require(!fs::is_symlink(pair.second), "output must not be a symbolic link");
    }
    YAML::Node report;
    report["applied"] = YAML::Node(YAML::NodeType::Sequence);
    std::set<std::string> ids, selections;
    for (const auto& target : targets) {
        const std::string id = target["id"].as<std::string>();
        require(ids.insert(id).second, "duplicate id " + id);
        const fs::path file = resolve(target["file"].as<std::string>());
        require(docs.count(file), "target file not included in scene: " + file.string());
        const std::string kind = target["kind"].as<std::string>();
        const std::string name = target["name"].as<std::string>();
        const std::string parameter = target["parameter"].as<std::string>();
        require(kind == "internal_part" || kind == "external_part" || kind == "actuator", "unsupported target kind");
        const bool scalar = parameter == "mass" || parameter == "max_setpoint";
        require(scalar || parameter == "compound_transform.xyz" || parameter == "compound_transform.rpy", "unsupported parameter");
        require((kind == "actuator") == (parameter == "max_setpoint"), "parameter not valid for target kind");
        std::vector<XMLElement*> matches;
        visit(docs[file]->RootElement(), [&](XMLElement* node) {
            if (node->Name() == kind && node->Attribute("name") && node->Attribute("name") == name) matches.push_back(node);
        });
        require(!matches.empty(), "target not found: " + id);
        require(target["occurrence"] || matches.size() == 1, "ambiguous name; specify zero-based occurrence: " + id);
        const int occurrence = target["occurrence"] ? target["occurrence"].as<int>() : 0;
        require(occurrence >= 0 && size_t(occurrence) < matches.size(), "invalid occurrence: " + id);
        require(selections.insert(file.string() + ":" + kind + ":" + name + ":" + std::to_string(occurrence) + ":" + parameter).second,
                "duplicate parameter target: " + id);
        XMLElement* element = matches[occurrence];
        if (kind == "actuator") require(element->Attribute("type") && std::string(element->Attribute("type")) == "thruster", "actuator must be a thruster");
        const char* tag = parameter == "mass" ? "mass" : parameter == "max_setpoint" ? "specs" : "compound_transform";
        const char* attr = parameter == "mass" ? "value" : parameter == "max_setpoint" ? "max_setpoint" : parameter == "compound_transform.xyz" ? "xyz" : "rpy";
        auto* child = element->FirstChildElement(tag);
        const size_t size = scalar ? 1 : 3;
        auto original = attributeNumbers(child, attr, size);
        auto low = numbers(target["min"], size), high = numbers(target["max"], size), delta = numbers(target["delta"], size);
        auto value = original;
        for (size_t i = 0; i < size; ++i) {
            require(low[i] <= high[i] && delta[i] >= low[i] && delta[i] <= high[i], "delta outside bounds: " + id);
            require(std::isfinite(original[i] + low[i]) && std::isfinite(original[i] + high[i]), "bounds overflow: " + id);
            if (scalar) require(original[i] + low[i] > 0, "mass and max_setpoint must stay positive: " + id);
            value[i] += delta[i];
        }
        child->SetAttribute(attr, format(value).c_str());
        YAML::Node entry;
        entry["id"] = id; entry["original"] = original; entry["delta"] = delta; entry["value"] = value;
        entry["parameter"] = parameter;
        // Resolve the XML selection to a runtime object. Stonefish parses external
        // parts before internal parts, irrespective of their XML interleaving.
        auto* parent = element->Parent() ? element->Parent()->ToElement() : nullptr;
        auto* robot = parent;
        while (robot && std::string(robot->Name()) != "robot")
            robot = robot->Parent() ? robot->Parent()->ToElement() : nullptr;
        if (robot && robot->Attribute("name")) {
            const std::string robotName = robot->Attribute("name");
            entry["runtime"]["robot"] = robotName;
            if (kind == "actuator") {
                entry["runtime"]["actuator"] = robotName + "/" + name;
            } else if (parent && parent->Attribute("name")) {
                const std::string link = robotName + "/" + parent->Attribute("name");
                entry["runtime"]["link"] = link;
                size_t index = 0, sameName = 0;
                bool found = false;
                for (const auto* partKind : {"external_part", "internal_part"}) {
                    for (auto* part = parent->FirstChildElement(partKind); part; part = part->NextSiblingElement(partKind)) {
                        if (part == element) { found = true; break; }
                        ++index;
                        if (part->Attribute("name") && name == part->Attribute("name")) ++sameName;
                    }
                    if (found) break;
                }
                entry["runtime"]["part_index"] = index;
                entry["runtime"]["part_name"] = link + "/" + name + (sameName ? std::to_string(sameName) : "");
            }
        }
        report["applied"].push_back(entry);
    }
    if (writeScenes) for (auto& pair : docs) {
        visit(pair.second->RootElement(), [&](XMLElement* node) {
            if (std::string(node->Name()) == "include") node->SetAttribute("file", outputs.at(resolve(node->Attribute("file"))).c_str());
        });
        require(pair.second->SaveFile(outputs.at(pair.first).c_str()) == tinyxml2::XML_SUCCESS, "cannot save generated scene");
    }
    report["scene_path"] = writeScenes ? outputs.at(main).string() : main.string();
    return report;
}
