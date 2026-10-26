#include <blockSQP2/structure.hpp>
#include <string>
#include <stdexcept>


std::string to_string(VariableGroupTypes vtype){
    switch (vtype){
        case VariableGroupTypes::control: return "control";
        case VariableGroupTypes::parameter: return "parameter";
        case VariableGroupTypes::initial_state: return "initial_state";
        case VariableGroupTypes::state: return "state";
    }
    return "";
}

VariableGroupTypes VariableGroupTypes_from_string(std::string_view vTypeName){
    if (vTypeName == to_string(VariableGroupTypes::control)) return VariableGroupTypes::control;
    if (vTypeName == to_string(VariableGroupTypes::parameter)) return VariableGroupTypes::parameter;
    if (vTypeName == to_string(VariableGroupTypes::initial_state)) return VariableGroupTypes::initial_state;
    if (vTypeName == to_string(VariableGroupTypes::state)) return VariableGroupTypes::state;
    
    throw std::invalid_argument(std::string("VariableGroupTypes_from_string: Name \"") + std::string(vTypeName) + std::string("\" does not match any available variable group type"));
}

VariableSystem::VariableSystem(std::unique_ptr<VariableGroupTypes[]> arg_vtypes, 
                               std::unique_ptr<std::unique_ptr<int[]>[]> arg_vgroup_ind, 
                               std::unique_ptr<int[]> arg_vgroup_n_ind,
                               int arg_n_vgroups
                               ): n_vgroups(arg_n_vgroups), vtypes(std::move(arg_vtypes)),
                                  vgroup_n_ind(std::move(arg_vgroup_n_ind)), vgroup_ind(std::move(arg_vgroup_ind)){}
                                  
VariableSystem::VariableSystem(int const *arg_vgroup_n_ind, int arg_n_vgroups): 
        n_vgroups(arg_n_vgroups), vtypes(std::make_unique<VariableGroupTypes[]>(arg_n_vgroups)),
        vgroup_n_ind(std::make_unique<int[]>(arg_n_vgroups)),
        vgroup_ind(std::make_unique<std::unique_ptr<int[]>[]>(arg_n_vgroups)){
    std::copy(arg_vgroup_n_ind, arg_vgroup_n_ind + n_vgroups, vgroup_n_ind.get());
    for (int i = 0; i < n_vgroups; i++) vgroup_ind[i] = std::make_unique<int[]>(arg_vgroup_n_ind[i]);
}
