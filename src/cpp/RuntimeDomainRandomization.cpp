#include "RuntimeDomainRandomization.h"
#include <Stonefish/core/SimulationManager.h>
#include <Stonefish/core/Robot.h>
#include <Stonefish/entities/solids/Compound.h>
#include <Stonefish/actuators/Thruster.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <stdexcept>

namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::invalid_argument("Runtime randomization: " + message);
}
std::vector<double> values(const YAML::Node& node, size_t n) {
    require(node.IsSequence() && node.size()==n, "wrong value vector length");
    std::vector<double> v;
    for (const auto& x : node) {
        auto value = x.as<double>();
        require(std::isfinite(value), "nonfinite value");
        v.push_back(value);
    }
    return v;
}
bool same(double a, double b) {
    return std::abs(a-b) <= 1e-12 * std::max(1., std::max(std::abs(a), std::abs(b)));
}
}

YAML::Node ApplyRuntimeDomainRandomization(sf::SimulationManager& sim, const YAML::Node& targets) {
    require(targets.IsSequence(), "targets must be a sequence");
    std::map<sf::Thruster*, sf::Scalar> limits;
#ifdef STONEFISH_COMPOUND_RESET_API
    std::map<sf::Compound*, std::map<size_t, sf::CompoundPartUpdate>> groups;
#endif
    std::set<std::string> selections;
    // Validate the complete request before mutating anything.
    for (const auto& target : targets) {
        const std::string parameter = target["parameter"].as<std::string>();
        const auto selection = target["runtime"];
        require(selection.IsMap(), "missing runtime selector; rebuild the XML editor");
        const std::string robotName = selection["robot"].as<std::string>();
        auto* robot = sim.getRobot(robotName);
        require(robot != nullptr, "robot not found: " + robotName);
        if (parameter == "max_setpoint") {
            const auto name = selection["actuator"].as<std::string>();
            auto* thruster = dynamic_cast<sf::Thruster*>(sim.getActuator(name));
            require(thruster != nullptr, "thruster not found: " + name);
            const auto v = values(target["value"], 1);
            require(v[0]>0, "max_setpoint must be positive");
            require(limits.emplace(thruster, v[0]).second, "duplicate thruster target");
            continue;
        }
        require(parameter=="mass" || parameter=="compound_transform.xyz" || parameter=="compound_transform.rpy", "unsupported parameter: " + parameter);
#ifndef STONEFISH_COMPOUND_RESET_API
        throw std::runtime_error("This Stonefish library lacks STONEFISH_COMPOUND_RESET_API; build with the supplied library patch");
#else
        const auto linkName = selection["link"].as<std::string>();
        auto* compound = dynamic_cast<sf::Compound*>(robot->getLink(linkName));
        require(compound != nullptr, "compound link not found: " + linkName);
        const int index = selection["part_index"].as<int>();
        require(index >= 0, "negative part index");
        const auto part = compound->getPart(size_t(index));
        require(part.solid && part.solid->getName() == selection["part_name"].as<std::string>(),
                "part identity differs from source scene; restart after editing scene structure");
        require(selections.insert(linkName + ":" + std::to_string(index) + ":" + parameter).second, "duplicate part parameter");
        auto& update = groups[compound][index];
        update.index = size_t(index);
        const auto v = values(target["value"], parameter=="mass" ? 1 : 3);
        if (parameter=="mass") {
            update.mass = v[0];
        } else {
            if (!update.origin) update.origin = part.origin;
            if (parameter=="compound_transform.xyz")
                update.origin->setOrigin(sf::Vector3(v[0],v[1],v[2]));
            else
                update.origin->setRotation(sf::Quaternion(v[2],v[1],v[0])); // XML RPY convention
        }
#endif
    }
    YAML::Node report;
    report["recalculated_compounds"] = 0;
    report["changed_thrusters"] = 0;
#ifdef STONEFISH_COMPOUND_RESET_API
    std::map<sf::Compound*, std::vector<sf::CompoundPartUpdate>> batches;
    for (const auto& group : groups) {
        for (const auto& update : group.second) batches[group.first].push_back(update.second);
        group.first->ValidatePartUpdates(batches[group.first], sim.getDynamicsWorld());
    }
    // All selector/bounds/structure errors have been checked before this point.
    // A numerical physics failure is fatal to this episode (no stale observation).
    for (const auto& batch : batches)
        if (batch.first->ApplyPartUpdates(batch.second, sim.getDynamicsWorld()))
            report["recalculated_compounds"] = report["recalculated_compounds"].as<int>() + 1;
#endif
    for (const auto& limit : limits) {
        if (!same(limit.first->getSetpointLimit(), limit.second)) {
            limit.first->setSetpointLimit(limit.second);
            // Setter updates limits; clear the previous episode's command as well.
            limit.first->setSetpoint(0);
            report["changed_thrusters"] = report["changed_thrusters"].as<int>() + 1;
        }
    }
    return report;
}
