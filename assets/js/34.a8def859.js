(window.webpackJsonp=window.webpackJsonp||[]).push([[34],{548:function(t,e,a){"use strict";a.r(e);var n=a(38),i=Object(n.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"builders-domain-scheduling-conditional-tasks"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#builders-domain-scheduling-conditional-tasks"}},[t._v("#")]),t._v(" builders.domain.scheduling.conditional_tasks")]),t._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[t._v("Domain specification")]),t._v(" "),a("skdecide-summary")],1),t._v(" "),a("h2",{attrs:{id:"withconditionaltasks"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#withconditionaltasks"}},[t._v("#")]),t._v(" WithConditionalTasks")]),t._v(" "),a("p",[t._v("A domain must inherit this class if some tasks only need be executed under some conditions\nand that the condition model can be expressed with Distribution objects.")]),t._v(" "),a("h3",{attrs:{id:"add-to-current-conditions"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#add-to-current-conditions"}},[t._v("#")]),t._v(" add_to_current_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"add_to_current_conditions",sig:{params:[{name:"self"},{name:"task",annotation:"int"},{name:"state"}]}}}),t._v(" "),a("p",[t._v("Samples completion conditions for a given task and add these conditions to the list of conditions in the\ngiven state. This function should be called when a task complete.")]),t._v(" "),a("h3",{attrs:{id:"get-all-condition-items"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-all-condition-items"}},[t._v("#")]),t._v(" get_all_condition_items "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_all_condition_items",sig:{params:[{name:"self"}],return:"Enum"}}}),t._v(" "),a("p",[t._v("Return an Enum with all the elements that can be used to define a condition.")]),t._v(" "),a("p",[t._v("Example:\nreturn\nConditionElementsExample(Enum):\nOK = 0\nNC_PART_1_OPERATION_1 = 1\nNC_PART_1_OPERATION_2 = 2\nNC_PART_2_OPERATION_1 = 3\nNC_PART_2_OPERATION_2 = 4\nHARDWARE_ISSUE_MACHINE_A = 5\nHARDWARE_ISSUE_MACHINE_B = 6")]),t._v(" "),a("h3",{attrs:{id:"get-all-unconditional-tasks"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-all-unconditional-tasks"}},[t._v("#")]),t._v(" get_all_unconditional_tasks "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_all_unconditional_tasks",sig:{params:[{name:"self"}],return:"set[int]"}}}),t._v(" "),a("p",[t._v("Returns the set of all task ids for which there are no conditions. These tasks are to be considered at\nthe start of a project (i.e. in the initial state).")]),t._v(" "),a("h3",{attrs:{id:"get-available-tasks"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-available-tasks"}},[t._v("#")]),t._v(" get_available_tasks "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_available_tasks",sig:{params:[{name:"self"},{name:"state"}],return:"set[int]"}}}),t._v(" "),a("p",[t._v("Returns the set of all task ids that can be considered under the conditions defined in the given state.\nNote that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks\nthat are remaining, or that have been completed, paused or started / resumed.")]),t._v(" "),a("h3",{attrs:{id:"get-task-existence-conditions"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-task-existence-conditions"}},[t._v("#")]),t._v(" get_task_existence_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_task_existence_conditions",sig:{params:[{name:"self"}],return:"dict[int, list[int]]"}}}),t._v(" "),a("p",[t._v("Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)\nfor the task to be part of the schedule. If a task has no entry in the dictionary,\nthere is no conditions for that task.")]),t._v(" "),a("p",[t._v("Example:\nreturn\n{\n20: [get_all_condition_items().NC_PART_1_OPERATION_1],\n21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]\n22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]\n}e")]),t._v(" "),a("h3",{attrs:{id:"get-task-on-completion-added-conditions"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-task-on-completion-added-conditions"}},[t._v("#")]),t._v(" get_task_on_completion_added_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_task_on_completion_added_conditions",sig:{params:[{name:"self"}],return:"dict[int, list[Distribution]]"}}}),t._v(" "),a("p",[t._v("Return a dict of list. The key of the dict is the task id and each list is composed of a list of tuples.\nEach tuple contains the probability (first item in tuple) that the conditionElement (second item in tuple)\nis True. The probabilities in the inner list should sum up to 1. The dictionary should only contains the keys\nof tasks that can create conditions.")]),t._v(" "),a("p",[t._v("Example:\nreturn\n{\n12:\n[\nDiscreteDistribution([(ConditionElementsExample.NC_PART_1_OPERATION_1, 0.1), (ConditionElementsExample.OK, 0.9)]),\nDiscreteDistribution([(ConditionElementsExample.HARDWARE_ISSUE_MACHINE_A, 0.05), ('paper', 0.1), (ConditionElementsExample.OK, 0.95)])\n]\n}")]),t._v(" "),a("h3",{attrs:{id:"sample-completion-conditions"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-completion-conditions"}},[t._v("#")]),t._v(" sample_completion_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"sample_completion_conditions",sig:{params:[{name:"self"},{name:"task",annotation:"int"}],return:"list[int]"}}}),t._v(" "),a("p",[t._v("Samples the condition distributions associated with the given task and return a list of sampled\nconditions.")]),t._v(" "),a("h3",{attrs:{id:"add-to-current-conditions-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#add-to-current-conditions-2"}},[t._v("#")]),t._v(" _add_to_current_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_add_to_current_conditions",sig:{params:[{name:"self"},{name:"task",annotation:"int"},{name:"state"}]}}}),t._v(" "),a("p",[t._v("Samples completion conditions for a given task and add these conditions to the list of conditions in the\ngiven state. This function should be called when a task complete.")]),t._v(" "),a("h3",{attrs:{id:"get-all-unconditional-tasks-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-all-unconditional-tasks-2"}},[t._v("#")]),t._v(" _get_all_unconditional_tasks "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_get_all_unconditional_tasks",sig:{params:[{name:"self"}],return:"set[int]"}}}),t._v(" "),a("p",[t._v("Returns the set of all task ids for which there are no conditions. These tasks are to be considered at\nthe start of a project (i.e. in the initial state).")]),t._v(" "),a("h3",{attrs:{id:"get-available-tasks-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-available-tasks-2"}},[t._v("#")]),t._v(" _get_available_tasks "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_get_available_tasks",sig:{params:[{name:"self"},{name:"state"}],return:"set[int]"}}}),t._v(" "),a("p",[t._v("Returns the set of all task ids that can be considered under the conditions defined in the given state.\nNote that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks\nthat are remaining, or that have been completed, paused or started / resumed.")]),t._v(" "),a("h3",{attrs:{id:"get-task-existence-conditions-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-task-existence-conditions-2"}},[t._v("#")]),t._v(" _get_task_existence_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_get_task_existence_conditions",sig:{params:[{name:"self"}],return:"dict[int, list[int]]"}}}),t._v(" "),a("p",[t._v("Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)\nfor the task to be part of the schedule. If a task has no entry in the dictionary,\nthere is no conditions for that task.")]),t._v(" "),a("p",[t._v("Example:\nreturn\n{\n20: [get_all_condition_items().NC_PART_1_OPERATION_1],\n21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]\n22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]\n}e")]),t._v(" "),a("h3",{attrs:{id:"sample-completion-conditions-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-completion-conditions-2"}},[t._v("#")]),t._v(" _sample_completion_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_sample_completion_conditions",sig:{params:[{name:"self"},{name:"task",annotation:"int"}],return:"list[int]"}}}),t._v(" "),a("p",[t._v("Samples the condition distributions associated with the given task and return a list of sampled\nconditions.")]),t._v(" "),a("h2",{attrs:{id:"withoutconditionaltasks"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#withoutconditionaltasks"}},[t._v("#")]),t._v(" WithoutConditionalTasks")]),t._v(" "),a("p",[t._v("A domain must inherit this class if all tasks need be executed without conditions.")]),t._v(" "),a("h3",{attrs:{id:"add-to-current-conditions-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#add-to-current-conditions-3"}},[t._v("#")]),t._v(" add_to_current_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"add_to_current_conditions",sig:{params:[{name:"self"},{name:"task",annotation:"int"},{name:"state"}]}}}),t._v(" "),a("p",[t._v("Samples completion conditions for a given task and add these conditions to the list of conditions in the\ngiven state. This function should be called when a task complete.")]),t._v(" "),a("h3",{attrs:{id:"get-all-condition-items-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-all-condition-items-2"}},[t._v("#")]),t._v(" get_all_condition_items "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_all_condition_items",sig:{params:[{name:"self"}],return:"Enum"}}}),t._v(" "),a("p",[t._v("Return an Enum with all the elements that can be used to define a condition.")]),t._v(" "),a("p",[t._v("Example:\nreturn\nConditionElementsExample(Enum):\nOK = 0\nNC_PART_1_OPERATION_1 = 1\nNC_PART_1_OPERATION_2 = 2\nNC_PART_2_OPERATION_1 = 3\nNC_PART_2_OPERATION_2 = 4\nHARDWARE_ISSUE_MACHINE_A = 5\nHARDWARE_ISSUE_MACHINE_B = 6")]),t._v(" "),a("h3",{attrs:{id:"get-all-unconditional-tasks-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-all-unconditional-tasks-3"}},[t._v("#")]),t._v(" get_all_unconditional_tasks "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_all_unconditional_tasks",sig:{params:[{name:"self"}],return:"set[int]"}}}),t._v(" "),a("p",[t._v("Returns the set of all task ids for which there are no conditions. These tasks are to be considered at\nthe start of a project (i.e. in the initial state).")]),t._v(" "),a("h3",{attrs:{id:"get-available-tasks-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-available-tasks-3"}},[t._v("#")]),t._v(" get_available_tasks "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_available_tasks",sig:{params:[{name:"self"},{name:"state"}],return:"set[int]"}}}),t._v(" "),a("p",[t._v("Returns the set of all task ids that can be considered under the conditions defined in the given state.\nNote that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks\nthat are remaining, or that have been completed, paused or started / resumed.")]),t._v(" "),a("h3",{attrs:{id:"get-task-existence-conditions-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-task-existence-conditions-3"}},[t._v("#")]),t._v(" get_task_existence_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_task_existence_conditions",sig:{params:[{name:"self"}],return:"dict[int, list[int]]"}}}),t._v(" "),a("p",[t._v("Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)\nfor the task to be part of the schedule. If a task has no entry in the dictionary,\nthere is no conditions for that task.")]),t._v(" "),a("p",[t._v("Example:\nreturn\n{\n20: [get_all_condition_items().NC_PART_1_OPERATION_1],\n21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]\n22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]\n}e")]),t._v(" "),a("h3",{attrs:{id:"get-task-on-completion-added-conditions-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-task-on-completion-added-conditions-2"}},[t._v("#")]),t._v(" get_task_on_completion_added_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_task_on_completion_added_conditions",sig:{params:[{name:"self"}],return:"dict[int, list[Distribution]]"}}}),t._v(" "),a("p",[t._v("Return a dict of list. The key of the dict is the task id and each list is composed of a list of tuples.\nEach tuple contains the probability (first item in tuple) that the conditionElement (second item in tuple)\nis True. The probabilities in the inner list should sum up to 1. The dictionary should only contains the keys\nof tasks that can create conditions.")]),t._v(" "),a("p",[t._v("Example:\nreturn\n{\n12:\n[\nDiscreteDistribution([(ConditionElementsExample.NC_PART_1_OPERATION_1, 0.1), (ConditionElementsExample.OK, 0.9)]),\nDiscreteDistribution([(ConditionElementsExample.HARDWARE_ISSUE_MACHINE_A, 0.05), ('paper', 0.1), (ConditionElementsExample.OK, 0.95)])\n]\n}")]),t._v(" "),a("h3",{attrs:{id:"sample-completion-conditions-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-completion-conditions-3"}},[t._v("#")]),t._v(" sample_completion_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"sample_completion_conditions",sig:{params:[{name:"self"},{name:"task",annotation:"int"}],return:"list[int]"}}}),t._v(" "),a("p",[t._v("Samples the condition distributions associated with the given task and return a list of sampled\nconditions.")]),t._v(" "),a("h3",{attrs:{id:"add-to-current-conditions-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#add-to-current-conditions-4"}},[t._v("#")]),t._v(" _add_to_current_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_add_to_current_conditions",sig:{params:[{name:"self"},{name:"task",annotation:"int"},{name:"state"}]}}}),t._v(" "),a("p",[t._v("Samples completion conditions for a given task and add these conditions to the list of conditions in the\ngiven state. This function should be called when a task complete.")]),t._v(" "),a("h3",{attrs:{id:"get-all-unconditional-tasks-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-all-unconditional-tasks-4"}},[t._v("#")]),t._v(" _get_all_unconditional_tasks "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_get_all_unconditional_tasks",sig:{params:[{name:"self"}],return:"set[int]"}}}),t._v(" "),a("p",[t._v("Returns the set of all task ids for which there are no conditions. These tasks are to be considered at\nthe start of a project (i.e. in the initial state).")]),t._v(" "),a("h3",{attrs:{id:"get-available-tasks-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-available-tasks-4"}},[t._v("#")]),t._v(" _get_available_tasks "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_get_available_tasks",sig:{params:[{name:"self"},{name:"state"}],return:"set[int]"}}}),t._v(" "),a("p",[t._v("Returns the set of all task ids that can be considered under the conditions defined in the given state.\nNote that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks\nthat are remaining, or that have been completed, paused or started / resumed.")]),t._v(" "),a("h3",{attrs:{id:"get-task-existence-conditions-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-task-existence-conditions-4"}},[t._v("#")]),t._v(" _get_task_existence_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_get_task_existence_conditions",sig:{params:[{name:"self"}],return:"dict[int, list[int]]"}}}),t._v(" "),a("p",[t._v("Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)\nfor the task to be part of the schedule. If a task has no entry in the dictionary,\nthere is no conditions for that task.")]),t._v(" "),a("p",[t._v("Example:\nreturn\n{\n20: [get_all_condition_items().NC_PART_1_OPERATION_1],\n21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]\n22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]\n}e")]),t._v(" "),a("h3",{attrs:{id:"sample-completion-conditions-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-completion-conditions-4"}},[t._v("#")]),t._v(" _sample_completion_conditions "),a("Badge",{attrs:{text:"WithConditionalTasks",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_sample_completion_conditions",sig:{params:[{name:"self"},{name:"task",annotation:"int"}],return:"list[int]"}}}),t._v(" "),a("p",[t._v("Samples the condition distributions associated with the given task and return a list of sampled\nconditions.")])],1)}),[],!1,null,null,null);e.default=i.exports}}]);