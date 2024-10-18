(window.webpackJsonp=window.webpackJsonp||[]).push([[105],{619:function(e,t,a){"use strict";a.r(t);var r=a(38),s=Object(r.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"hub-solver-do-solver-sgs-policies"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#hub-solver-do-solver-sgs-policies"}},[e._v("#")]),e._v(" hub.solver.do_solver.sgs_policies")]),e._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[e._v("Domain specification")]),e._v(" "),a("skdecide-summary")],1),e._v(" "),a("h2",{attrs:{id:"basepolicymethod"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#basepolicymethod"}},[e._v("#")]),e._v(" BasePolicyMethod")]),e._v(" "),a("p",[e._v("Base options to define Scheduling policies")]),e._v(" "),a("h3",{attrs:{id:"follow-gantt"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#follow-gantt"}},[e._v("#")]),e._v(" FOLLOW_GANTT "),a("Badge",{attrs:{text:"BasePolicyMethod",type:"tip"}})],1),e._v(" "),a("p",[e._v("Strictly return scheduling policy based on the gantt chart.\nBased on the time stored in the state, task are started at the right time.")]),e._v(" "),a("h3",{attrs:{id:"sgs-index-freedom"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sgs-index-freedom"}},[e._v("#")]),e._v(" SGS_INDEX_FREEDOM "),a("Badge",{attrs:{text:"BasePolicyMethod",type:"tip"}})],1),e._v(" "),a("p",[e._v('At a given state, look for the first task "TASK" in the permutation that is not started or scheduled yet,\nIf it\'s not available to start yet, some other task are considered candidates based their "ordering"\ncloseness to the starting time of "TASK", the policy will consider starting task that are close to\nthe one that was first expected. '),a("code",[e._v("delta_index_freedom")]),e._v(" is the parameter that impacts this setting.")]),e._v(" "),a("h3",{attrs:{id:"sgs-precedence"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sgs-precedence"}},[e._v("#")]),e._v(" SGS_PRECEDENCE "),a("Badge",{attrs:{text:"BasePolicyMethod",type:"tip"}})],1),e._v(" "),a("p",[e._v("At a given state, look for the first available task\nin an ordered permutation that is start-able and do it.\nIf no activity is launchable, just advance in time.")]),e._v(" "),a("h3",{attrs:{id:"sgs-ready"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sgs-ready"}},[e._v("#")]),e._v(" SGS_READY "),a("Badge",{attrs:{text:"BasePolicyMethod",type:"tip"}})],1),e._v(" "),a("p",[e._v("Same as SGS_PRECEDENCE, one of those 2 will be in deprecation soon.")]),e._v(" "),a("h3",{attrs:{id:"sgs-strict"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sgs-strict"}},[e._v("#")]),e._v(" SGS_STRICT "),a("Badge",{attrs:{text:"BasePolicyMethod",type:"tip"}})],1),e._v(" "),a("p",[e._v("At a given state, look for the first task in the permutation that is not started or scheduled yet.\nIf it's not available to start yet, we advance in time until it is.")]),e._v(" "),a("div",{staticClass:"custom-block warning"},[a("p",{staticClass:"custom-block-title"},[e._v("WARNING")]),e._v(" "),a("p",[e._v("This will only work when the permutation of tasks fulfills the precedence constraints.")])]),e._v(" "),a("h3",{attrs:{id:"sgs-time-freedom"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sgs-time-freedom"}},[e._v("#")]),e._v(" SGS_TIME_FREEDOM "),a("Badge",{attrs:{text:"BasePolicyMethod",type:"tip"}})],1),e._v(" "),a("p",[e._v('At a given state, look for the first task "TASK" in the permutation that is not started or scheduled yet,\nIf it\'s not available to start yet, some other task are considered candidates based their time\ncloseness to the starting time of "TASK", the policy will consider starting task that are close to\nthe one that was first expected. '),a("code",[e._v("delta_time_freedom")]),e._v(" is the parameter that impacts this setting.")]),e._v(" "),a("h2",{attrs:{id:"policymethodparams"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#policymethodparams"}},[e._v("#")]),e._v(" PolicyMethodParams")]),e._v(" "),a("p",[e._v("Wrapped params for scheduling policy parameters, see BasePolicyMethod for more details")]),e._v(" "),a("h3",{attrs:{id:"constructor"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#constructor"}},[e._v("#")]),e._v(" Constructor "),a("Badge",{attrs:{text:"PolicyMethodParams",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"PolicyMethodParams",sig:{params:[{name:"base_policy_method",annotation:"BasePolicyMethod"},{name:"delta_time_freedom",default:"10"},{name:"delta_index_freedom",default:"10"}]}}}),e._v(" "),a("p",[e._v("Initialize self.  See help(type(self)) for accurate signature.")]),e._v(" "),a("h3",{attrs:{id:"complete-with-default-hyperparameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#complete-with-default-hyperparameters"}},[e._v("#")]),e._v(" complete_with_default_hyperparameters "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"complete_with_default_hyperparameters",sig:{params:[{name:"kwargs",annotation:"dict[str, Any]"},{name:"names",default:"None",annotation:"Optional[list[str]]"}]}}}),e._v(" "),a("p",[e._v("Add missing hyperparameters to kwargs by using default values")]),e._v(" "),a("p",[e._v("Args:\nkwargs: keyword arguments to complete (e.g. for "),a("code",[e._v("__init__")]),e._v(", "),a("code",[e._v("init_model")]),e._v(", or "),a("code",[e._v("solve")]),e._v(")\nnames: names of the hyperparameters to add if missing.\nBy default, all available hyperparameters.")]),e._v(" "),a("p",[e._v("Returns:\na new dictionary, completion of kwargs")]),e._v(" "),a("h3",{attrs:{id:"copy-and-update-hyperparameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#copy-and-update-hyperparameters"}},[e._v("#")]),e._v(" copy_and_update_hyperparameters "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"copy_and_update_hyperparameters",sig:{params:[{name:"names",default:"None",annotation:"Optional[list[str]]"},{name:"**kwargs_by_name",annotation:"dict[str, Any]"}],return:"list[Hyperparameter]"}}}),e._v(" "),a("p",[e._v("Copy hyperparameters definition of this class and update them with specified kwargs.")]),e._v(" "),a("p",[e._v("This is useful to define hyperparameters for a child class\nfor which only choices of the hyperparameter change for instance.")]),e._v(" "),a("p",[e._v("Args:\nnames: names of hyperparameters to copy. Default to all.\n**kwargs_by_name: for each hyperparameter specified by its name,\nthe attributes to update. If a given hyperparameter name is not specified,\nthe hyperparameter is copied without further update.")]),e._v(" "),a("p",[e._v("Returns:")]),e._v(" "),a("h3",{attrs:{id:"get-default-hyperparameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-default-hyperparameters"}},[e._v("#")]),e._v(" get_default_hyperparameters "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_default_hyperparameters",sig:{params:[{name:"names",default:"None",annotation:"Optional[list[str]]"}],return:"dict[str, Any]"}}}),e._v(" "),a("p",[e._v("Get hyperparameters default values.")]),e._v(" "),a("p",[e._v("Args:\nnames: names of the hyperparameters to choose.\nBy default, all available hyperparameters will be suggested.")]),e._v(" "),a("p",[e._v("Returns:\na mapping between hyperparameter's name_in_kwargs and its default value (None if not specified)")]),e._v(" "),a("h3",{attrs:{id:"get-hyperparameter"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-hyperparameter"}},[e._v("#")]),e._v(" get_hyperparameter "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_hyperparameter",sig:{params:[{name:"name",annotation:"str"}],return:"Hyperparameter"}}}),e._v(" "),a("p",[e._v("Get hyperparameter from given name.")]),e._v(" "),a("h3",{attrs:{id:"get-hyperparameters-by-name"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-hyperparameters-by-name"}},[e._v("#")]),e._v(" get_hyperparameters_by_name "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_hyperparameters_by_name",sig:{params:[],return:"dict[str, Hyperparameter]"}}}),e._v(" "),a("p",[e._v("Mapping from name to corresponding hyperparameter.")]),e._v(" "),a("h3",{attrs:{id:"get-hyperparameters-names"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-hyperparameters-names"}},[e._v("#")]),e._v(" get_hyperparameters_names "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_hyperparameters_names",sig:{params:[],return:"list[str]"}}}),e._v(" "),a("p",[e._v("List of hyperparameters names.")]),e._v(" "),a("h3",{attrs:{id:"suggest-hyperparameter-with-optuna"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#suggest-hyperparameter-with-optuna"}},[e._v("#")]),e._v(" suggest_hyperparameter_with_optuna "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"suggest_hyperparameter_with_optuna",sig:{params:[{name:"trial",annotation:"optuna.trial.Trial"},{name:"name",annotation:"str"},{name:"prefix",default:"",annotation:"str"},{name:"**kwargs"}],return:"Any"}}}),e._v(" "),a("p",[e._v("Suggest hyperparameter value during an Optuna trial.")]),e._v(" "),a("p",[e._v("This can be used during Optuna hyperparameters tuning.")]),e._v(" "),a("p",[e._v("Args:\ntrial: optuna trial during hyperparameters tuning\nname: name of the hyperparameter to choose\nprefix: prefix to add to optuna corresponding parameter name\n(useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)\n**kwargs: options for optuna hyperparameter suggestions")]),e._v(" "),a("p",[e._v("Returns:")]),e._v(" "),a("p",[e._v("kwargs can be used to pass relevant arguments to")]),e._v(" "),a("ul",[a("li",[e._v("trial.suggest_float()")]),e._v(" "),a("li",[e._v("trial.suggest_int()")]),e._v(" "),a("li",[e._v("trial.suggest_categorical()")])]),e._v(" "),a("p",[e._v("For instance it can")]),e._v(" "),a("ul",[a("li",[e._v("add a low/high value if not existing for the hyperparameter\nor override it to narrow the search. (for float or int hyperparameters)")]),e._v(" "),a("li",[e._v("add a step or log argument (for float or int hyperparameters,\nsee optuna.trial.Trial.suggest_float())")]),e._v(" "),a("li",[e._v("override choices for categorical or enum parameters to narrow the search")])]),e._v(" "),a("h3",{attrs:{id:"suggest-hyperparameters-with-optuna"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#suggest-hyperparameters-with-optuna"}},[e._v("#")]),e._v(" suggest_hyperparameters_with_optuna "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"suggest_hyperparameters_with_optuna",sig:{params:[{name:"trial",annotation:"optuna.trial.Trial"},{name:"names",default:"None",annotation:"Optional[list[str]]"},{name:"kwargs_by_name",default:"None",annotation:"Optional[dict[str, dict[str, Any]]]"},{name:"fixed_hyperparameters",default:"None",annotation:"Optional[dict[str, Any]]"},{name:"prefix",default:"",annotation:"str"}],return:"dict[str, Any]"}}}),e._v(" "),a("p",[e._v("Suggest hyperparameters values during an Optuna trial.")]),e._v(" "),a("p",[e._v("Args:\ntrial: optuna trial during hyperparameters tuning\nnames: names of the hyperparameters to choose.\nBy default, all available hyperparameters will be suggested.\nIf "),a("code",[e._v("fixed_hyperparameters")]),e._v(" is provided, the corresponding names are removed from "),a("code",[e._v("names")]),e._v(".\nkwargs_by_name: options for optuna hyperparameter suggestions, by hyperparameter name\nfixed_hyperparameters: values of fixed hyperparameters, useful for suggesting subbrick hyperparameters,\nif the subbrick class is not suggested by this method, but already fixed.\nWill be added to the suggested hyperparameters.\nprefix: prefix to add to optuna corresponding parameters\n(useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)")]),e._v(" "),a("p",[e._v("Returns:\nmapping between the hyperparameter name and its suggested value.\nIf the hyperparameter has an attribute "),a("code",[e._v("name_in_kwargs")]),e._v(", this is used as the key in the mapping\ninstead of the actual hyperparameter name.\nthe mapping is updated with "),a("code",[e._v("fixed_hyperparameters")]),e._v(".")]),e._v(" "),a("p",[e._v("kwargs_by_name[some_name] will be passed as **kwargs to suggest_hyperparameter_with_optuna(name=some_name)")]),e._v(" "),a("h2",{attrs:{id:"policyrcpsp"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#policyrcpsp"}},[e._v("#")]),e._v(" PolicyRCPSP")]),e._v(" "),a("p",[e._v("Policy object containing results of scheduling solver policy.")]),e._v(" "),a("h4",{attrs:{id:"attributes"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#attributes"}},[e._v("#")]),e._v(" Attributes")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("domain")]),e._v(": scheduling domain where the policy will be applied")]),e._v(" "),a("li",[a("strong",[e._v("policy_method_params")]),e._v(": params of the policy")]),e._v(" "),a("li",[a("strong",[e._v("permutation_task")]),e._v(": list of tasks ids, representing a priority list for scheduling")]),e._v(" "),a("li",[a("strong",[e._v("modes_dictionnary")]),e._v(": when relevant (multimode rcpsp for e.g) specifies in which mode a task is executed")]),e._v(" "),a("li",[a("strong",[e._v("schedule")]),e._v(": when given, details the schedule to follow : this will be relevant for deterministic scheduling problems")]),e._v(" "),a("li",[a("strong",[e._v("resource_allocation")]),e._v(": when relevant (multiskill problems for e.g), list the allocated (unitary) resources to the tasks")]),e._v(" "),a("li",[a("strong",[e._v("resource_allocation_priority")]),e._v(": for each task, store a preference order for resources to be allocated to the task.\nResource will be greedily allocated based on this priority")])]),e._v(" "),a("h3",{attrs:{id:"constructor-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#constructor-2"}},[e._v("#")]),e._v(" Constructor "),a("Badge",{attrs:{text:"PolicyRCPSP",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"PolicyRCPSP",sig:{params:[{name:"domain",annotation:"SchedulingDomain"},{name:"policy_method_params",annotation:"PolicyMethodParams"},{name:"permutation_task",annotation:"list[int]"},{name:"modes_dictionnary",annotation:"dict[int, int]"},{name:"schedule",default:"None",annotation:"Optional[dict[int, dict[str, int]]]"},{name:"resource_allocation",default:"None",annotation:"Optional[dict[int, list[str]]]"},{name:"resource_allocation_priority",default:"None",annotation:"Optional[dict[int, list[str]]]"}]}}}),e._v(" "),a("p",[e._v("Initialize self.  See help(type(self)) for accurate signature.")]),e._v(" "),a("h3",{attrs:{id:"get-next-action"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action"}},[e._v("#")]),e._v(" get_next_action "),a("Badge",{attrs:{text:"DeterministicPolicies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_next_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),e._v(" "),a("p",[e._v("Get the next deterministic action (from the solver's current policy).")]),e._v(" "),a("h4",{attrs:{id:"parameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation for which next action is requested.")])]),e._v(" "),a("h4",{attrs:{id:"returns"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The next deterministic action.")]),e._v(" "),a("h3",{attrs:{id:"get-next-action-distribution"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-distribution"}},[e._v("#")]),e._v(" get_next_action_distribution "),a("Badge",{attrs:{text:"UncertainPolicies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_next_action_distribution",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"Distribution[D.T_agent[D.T_concurrency[D.T_event]]]"}}}),e._v(" "),a("p",[e._v("Get the probabilistic distribution of next action for the given observation (from the solver's current\npolicy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-2"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation to consider.")])]),e._v(" "),a("h4",{attrs:{id:"returns-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-2"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The probabilistic distribution of next action.")]),e._v(" "),a("h3",{attrs:{id:"is-policy-defined-for"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-policy-defined-for"}},[e._v("#")]),e._v(" is_policy_defined_for "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"is_policy_defined_for",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"bool"}}}),e._v(" "),a("p",[e._v("Check whether the solver's current policy is defined for the given observation.")]),e._v(" "),a("h4",{attrs:{id:"parameters-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-3"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation to consider.")])]),e._v(" "),a("h4",{attrs:{id:"returns-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-3"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("True if the policy is defined for the given observation memory (False otherwise).")]),e._v(" "),a("h3",{attrs:{id:"sample-action"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-action"}},[e._v("#")]),e._v(" sample_action "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"sample_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),e._v(" "),a("p",[e._v("Sample an action for the given observation (from the solver's current policy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-4"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation for which an action must be sampled.")])]),e._v(" "),a("h4",{attrs:{id:"returns-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-4"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The sampled action.")]),e._v(" "),a("h3",{attrs:{id:"get-next-action-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-2"}},[e._v("#")]),e._v(" _get_next_action "),a("Badge",{attrs:{text:"DeterministicPolicies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_get_next_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),e._v(" "),a("p",[e._v("Get the next deterministic action (from the solver's current policy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-5"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-5"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation for which next action is requested.")])]),e._v(" "),a("h4",{attrs:{id:"returns-5"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-5"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The next deterministic action.")]),e._v(" "),a("h3",{attrs:{id:"get-next-action-distribution-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-distribution-2"}},[e._v("#")]),e._v(" _get_next_action_distribution "),a("Badge",{attrs:{text:"UncertainPolicies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_get_next_action_distribution",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"Distribution[D.T_agent[D.T_concurrency[D.T_event]]]"}}}),e._v(" "),a("p",[e._v("Get the probabilistic distribution of next action for the given observation (from the solver's current\npolicy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-6"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-6"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation to consider.")])]),e._v(" "),a("h4",{attrs:{id:"returns-6"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-6"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The probabilistic distribution of next action.")]),e._v(" "),a("h3",{attrs:{id:"is-policy-defined-for-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-policy-defined-for-2"}},[e._v("#")]),e._v(" _is_policy_defined_for "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_is_policy_defined_for",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"bool"}}}),e._v(" "),a("p",[e._v("Check whether the solver's current policy is defined for the given observation.")]),e._v(" "),a("h4",{attrs:{id:"parameters-7"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-7"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation to consider.")])]),e._v(" "),a("h4",{attrs:{id:"returns-7"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-7"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("True if the policy is defined for the given observation memory (False otherwise).")]),e._v(" "),a("h3",{attrs:{id:"sample-action-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-action-2"}},[e._v("#")]),e._v(" _sample_action "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_sample_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),e._v(" "),a("p",[e._v("Sample an action for the given observation (from the solver's current policy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-8"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-8"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation for which an action must be sampled.")])]),e._v(" "),a("h4",{attrs:{id:"returns-8"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-8"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The sampled action.")]),e._v(" "),a("h2",{attrs:{id:"next-action-follow-static-gantt"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#next-action-follow-static-gantt"}},[e._v("#")]),e._v(" next_action_follow_static_gantt")]),e._v(" "),a("skdecide-signature",{attrs:{name:"next_action_follow_static_gantt",sig:{params:[{name:"policy_rcpsp",annotation:"PolicyRCPSP"},{name:"state",annotation:"State"},{name:"check_if_applicable",default:"False",annotation:"bool"},{name:"**kwargs"}]}}}),e._v(" "),a("p",[e._v("Implements the policy with the parameters FOLLOW_GANTT (see its doc)")]),e._v(" "),a("h2",{attrs:{id:"next-action-sgs-first-task-precedence-ready"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#next-action-sgs-first-task-precedence-ready"}},[e._v("#")]),e._v(" next_action_sgs_first_task_precedence_ready")]),e._v(" "),a("skdecide-signature",{attrs:{name:"next_action_sgs_first_task_precedence_ready",sig:{params:[{name:"policy_rcpsp",annotation:"PolicyRCPSP"},{name:"state",annotation:"State"},{name:"check_if_applicable",default:"False",annotation:"bool"},{name:"**kwargs"}]}}}),e._v(" "),a("p",[e._v("Implements the policy with the parameters SGS_PRECEDENCE (see its doc)")]),e._v(" "),a("h2",{attrs:{id:"next-action-sgs-first-task-ready"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#next-action-sgs-first-task-ready"}},[e._v("#")]),e._v(" next_action_sgs_first_task_ready")]),e._v(" "),a("skdecide-signature",{attrs:{name:"next_action_sgs_first_task_ready",sig:{params:[{name:"policy_rcpsp",annotation:"PolicyRCPSP"},{name:"state",annotation:"State"},{name:"check_if_applicable",default:"False",annotation:"bool"},{name:"domain_sk_decide",default:"None",annotation:"Union[MultiModeRCPSP, SingleModeRCPSP]"},{name:"**kwargs"}]}}}),e._v(" "),a("p",[e._v("Implements the policy with the parameters SGS_READY (see its doc)")]),e._v(" "),a("h2",{attrs:{id:"next-action-sgs-strict"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#next-action-sgs-strict"}},[e._v("#")]),e._v(" next_action_sgs_strict")]),e._v(" "),a("skdecide-signature",{attrs:{name:"next_action_sgs_strict",sig:{params:[{name:"policy_rcpsp",annotation:"PolicyRCPSP"},{name:"state",annotation:"State"},{name:"check_if_applicable",default:"False",annotation:"bool"},{name:"domain_sk_decide",default:"None",annotation:"Union[MultiModeRCPSP, SingleModeRCPSP]"},{name:"**kwargs"}]}}}),e._v(" "),a("p",[e._v("Implements the policy with the parameters SGS_STRICT (see its doc)")]),e._v(" "),a("h2",{attrs:{id:"next-action-sgs-time-freedom"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#next-action-sgs-time-freedom"}},[e._v("#")]),e._v(" next_action_sgs_time_freedom")]),e._v(" "),a("skdecide-signature",{attrs:{name:"next_action_sgs_time_freedom",sig:{params:[{name:"policy_rcpsp",annotation:"PolicyRCPSP"},{name:"state",annotation:"State"},{name:"check_if_applicable",default:"False",annotation:"bool"},{name:"domain_sk_decide",default:"None",annotation:"Union[MultiModeRCPSP, SingleModeRCPSP]"},{name:"delta_time_freedom",default:"10",annotation:"int"},{name:"**kwargs"}]}}}),e._v(" "),a("p",[e._v("Implements the policy with the parameters SGS_TIME_FREEDOM (see its doc)")]),e._v(" "),a("h2",{attrs:{id:"next-action-sgs-index-freedom"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#next-action-sgs-index-freedom"}},[e._v("#")]),e._v(" next_action_sgs_index_freedom")]),e._v(" "),a("skdecide-signature",{attrs:{name:"next_action_sgs_index_freedom",sig:{params:[{name:"policy_rcpsp",annotation:"PolicyRCPSP"},{name:"state",annotation:"State"},{name:"check_if_applicable",default:"False",annotation:"bool"},{name:"domain_sk_decide",default:"None",annotation:"Union[MultiModeRCPSP, SingleModeRCPSP]"},{name:"delta_index_freedom",default:"10",annotation:"int"},{name:"**kwargs"}]}}}),e._v(" "),a("p",[e._v("Implements the policy with the parameters SGS_INDEX_FREEDOM (see its doc)")])],1)}),[],!1,null,null,null);t.default=s.exports}}]);