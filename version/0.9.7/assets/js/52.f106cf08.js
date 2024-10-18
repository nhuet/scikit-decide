(window.webpackJsonp=window.webpackJsonp||[]).push([[52],{565:function(a,e,t){"use strict";t.r(e);var s=t(38),r=Object(s.a)({},(function(){var a=this,e=a.$createElement,t=a._self._c||e;return t("ContentSlotsDistributor",{attrs:{"slot-key":a.$parent.slotKey}},[t("h1",{attrs:{id:"builders-domain-value"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#builders-domain-value"}},[a._v("#")]),a._v(" builders.domain.value")]),a._v(" "),t("div",{staticClass:"custom-block tip"},[t("p",{staticClass:"custom-block-title"},[a._v("Domain specification")]),a._v(" "),t("skdecide-summary")],1),a._v(" "),t("h2",{attrs:{id:"rewards"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#rewards"}},[a._v("#")]),a._v(" Rewards")]),a._v(" "),t("p",[a._v("A domain must inherit this class if it sends rewards (positive and/or negative).")]),a._v(" "),t("h3",{attrs:{id:"check-value"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#check-value"}},[a._v("#")]),a._v(" check_value "),t("Badge",{attrs:{text:"Rewards",type:"tip"}})],1),a._v(" "),t("skdecide-signature",{attrs:{name:"check_value",sig:{params:[{name:"self"},{name:"value",annotation:"Value[D.T_value]"}],return:"bool"}}}),a._v(" "),t("p",[a._v("Check that a value is compliant with its reward specification.")]),a._v(" "),t("div",{staticClass:"custom-block tip"},[t("p",{staticClass:"custom-block-title"},[a._v("TIP")]),a._v(" "),t("p",[a._v("This function returns always True by default because any kind of reward should be accepted at this level.")])]),a._v(" "),t("h4",{attrs:{id:"parameters"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#parameters"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),t("ul",[t("li",[t("strong",[a._v("value")]),a._v(": The value to check.")])]),a._v(" "),t("h4",{attrs:{id:"returns"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#returns"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),t("p",[a._v("True if the value is compliant (False otherwise).")]),a._v(" "),t("h3",{attrs:{id:"check-value-2"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#check-value-2"}},[a._v("#")]),a._v(" _check_value "),t("Badge",{attrs:{text:"Rewards",type:"tip"}})],1),a._v(" "),t("skdecide-signature",{attrs:{name:"_check_value",sig:{params:[{name:"self"},{name:"value",annotation:"Value[D.T_value]"}],return:"bool"}}}),a._v(" "),t("p",[a._v("Check that a value is compliant with its reward specification.")]),a._v(" "),t("div",{staticClass:"custom-block tip"},[t("p",{staticClass:"custom-block-title"},[a._v("TIP")]),a._v(" "),t("p",[a._v("This function returns always True by default because any kind of reward should be accepted at this level.")])]),a._v(" "),t("h4",{attrs:{id:"parameters-2"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#parameters-2"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),t("ul",[t("li",[t("strong",[a._v("value")]),a._v(": The value to check.")])]),a._v(" "),t("h4",{attrs:{id:"returns-2"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#returns-2"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),t("p",[a._v("True if the value is compliant (False otherwise).")]),a._v(" "),t("h2",{attrs:{id:"positivecosts"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#positivecosts"}},[a._v("#")]),a._v(" PositiveCosts")]),a._v(" "),t("p",[a._v("A domain must inherit this class if it sends only positive costs (i.e. negative rewards).")]),a._v(" "),t("p",[a._v("Having only positive costs is a required assumption for certain solvers to work, such as classical planners.")]),a._v(" "),t("h3",{attrs:{id:"check-value-3"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#check-value-3"}},[a._v("#")]),a._v(" check_value "),t("Badge",{attrs:{text:"Rewards",type:"warn"}})],1),a._v(" "),t("skdecide-signature",{attrs:{name:"check_value",sig:{params:[{name:"self"},{name:"value",annotation:"Value[D.T_value]"}],return:"bool"}}}),a._v(" "),t("p",[a._v("Check that a value is compliant with its reward specification.")]),a._v(" "),t("div",{staticClass:"custom-block tip"},[t("p",{staticClass:"custom-block-title"},[a._v("TIP")]),a._v(" "),t("p",[a._v("This function returns always True by default because any kind of reward should be accepted at this level.")])]),a._v(" "),t("h4",{attrs:{id:"parameters-3"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#parameters-3"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),t("ul",[t("li",[t("strong",[a._v("value")]),a._v(": The value to check.")])]),a._v(" "),t("h4",{attrs:{id:"returns-3"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#returns-3"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),t("p",[a._v("True if the value is compliant (False otherwise).")]),a._v(" "),t("h3",{attrs:{id:"check-value-4"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#check-value-4"}},[a._v("#")]),a._v(" _check_value "),t("Badge",{attrs:{text:"Rewards",type:"warn"}})],1),a._v(" "),t("skdecide-signature",{attrs:{name:"_check_value",sig:{params:[{name:"self"},{name:"value",annotation:"Value[D.T_value]"}],return:"bool"}}}),a._v(" "),t("p",[a._v("Check that a value is compliant with its cost specification (must be positive).")]),a._v(" "),t("div",{staticClass:"custom-block tip"},[t("p",{staticClass:"custom-block-title"},[a._v("TIP")]),a._v(" "),t("p",[a._v("This function calls "),t("code",[a._v("PositiveCost._is_positive()")]),a._v(" to determine if a value is positive (can be overridden for\nadvanced value types).")])]),a._v(" "),t("h4",{attrs:{id:"parameters-4"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#parameters-4"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),t("ul",[t("li",[t("strong",[a._v("value")]),a._v(": The value to check.")])]),a._v(" "),t("h4",{attrs:{id:"returns-4"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#returns-4"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),t("p",[a._v("True if the value is compliant (False otherwise).")]),a._v(" "),t("h3",{attrs:{id:"is-positive"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#is-positive"}},[a._v("#")]),a._v(" _is_positive "),t("Badge",{attrs:{text:"PositiveCosts",type:"tip"}})],1),a._v(" "),t("skdecide-signature",{attrs:{name:"_is_positive",sig:{params:[{name:"self"},{name:"cost",annotation:"D.T_value"}],return:"bool"}}}),a._v(" "),t("p",[a._v("Determine if a value is positive (can be overridden for advanced value types).")]),a._v(" "),t("h4",{attrs:{id:"parameters-5"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#parameters-5"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),t("ul",[t("li",[t("strong",[a._v("cost")]),a._v(": The cost to evaluate.")])]),a._v(" "),t("h4",{attrs:{id:"returns-5"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#returns-5"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),t("p",[a._v("True if the cost is positive (False otherwise).")])],1)}),[],!1,null,null,null);e.default=r.exports}}]);