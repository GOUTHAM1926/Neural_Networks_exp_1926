#include "ProcessGroupNCCL.h"
// #include "Error_logs.hpp"

#include <memory>
#include <string>
#include <unordered_map>

inline int global_counter = 0;


inline std::string get_group_name(){
	return std::to_string(global_counter++); 
}

class GroupRegister{
public:
	// GroupRegister(const std::string& group_name, std::shared_ptr<ProcessGroupNCCL> process_group);
	GroupRegister() = default;

	inline void register_group(const std::string& group_name, std::shared_ptr<ProcessGroupNCCL> process_group){

		auto it = name_to_group.find(group_name);

		// CONDITION_ASSERT(it == name_to_group.end(),
		// 		"Group Already present");

		name_to_group[group_name] = process_group;
	}
	
	inline void unregister_group(const std::string& group_name){
		
		name_to_group.erase(group_name);
	}

	inline std::shared_ptr<ProcessGroupNCCL> resolve_process_group(const std::string& group_name){
		auto it = name_to_group.find(group_name);

		// CONDITION_ASSERT_TRUE(it != name_to_group.end(),
		// 		"Group not found!!");

		auto group = it->second;

		// CONDITION_ASSERT_TRUE(group == nullptr, 
		// 		"GROUP is either destroyed or no group with the name");

		return group;
	}
	inline void print_groups(){
		
		for(auto [name,_] : name_to_group){
			std::cout << name << std::endl;
		}
	}

private:
	std::unordered_map<std::string, std::shared_ptr<ProcessGroupNCCL>> name_to_group;
};




inline GroupRegister* process_registry;


inline void register_group(const std::string& group_name, std::shared_ptr<ProcessGroupNCCL> process_group){
	if(process_registry == nullptr){
		process_registry = new GroupRegister();
	}
	process_registry->register_group(group_name, process_group);
}

inline void unregister_group(const std::string& group_name){
	process_registry->unregister_group(group_name);
}

inline void resolve_process_group(const std::string& group_name){
	process_registry->resolve_process_group(group_name);
}
