#ifndef BLOCKSQP2_STRUCTURE_HPP
#define BLOCKSQP2_STRUCTURE_HPP

#include <memory>
#include <string>

enum class VariableGroupTypes{
    control,
    parameter,
    initial_state,
    state,
};

inline bool is_free(VariableGroupTypes vtype){return vtype != VariableGroupTypes::state;}

std::string to_string(VariableGroupTypes vtype);
VariableGroupTypes VariableGroupTypes_from_string(std::string_view vTypeName);


class VariableSystem{
    public:
    int n_vgroups;
    std::unique_ptr<VariableGroupTypes[]> vtypes;
    std::unique_ptr<int[]> vgroup_n_ind;
    std::unique_ptr<std::unique_ptr<int[]>[]> vgroup_ind;
    
    VariableSystem(std::unique_ptr<VariableGroupTypes[]> arg_vtypes, 
                   std::unique_ptr<std::unique_ptr<int[]>[]> arg_vgroup_ind, 
                   std::unique_ptr<int[]> arg_vgroup_n_ind,
                   int arg_n_vgroups
                   );
    VariableSystem(int const *arg_vgroup_n_ind, int arg_n_vgroups);
};

#endif