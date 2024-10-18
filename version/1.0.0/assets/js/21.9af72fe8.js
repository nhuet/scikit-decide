(window.webpackJsonp=window.webpackJsonp||[]).push([[21],{536:function(t,e,a){"use strict";a.r(e);var r=a(38),o=Object(r.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"notebooks"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#notebooks"}},[t._v("#")]),t._v(" Notebooks")]),t._v(" "),a("p",[t._v("We present here a curated list of notebooks recommended to start with scikit-decide, available in the "),a("code",[t._v("notebooks/")]),t._v(" folder of the repository.")]),t._v(" "),a("p"),a("div",{staticClass:"table-of-contents"},[a("ul",[a("li",[a("a",{attrs:{href:"#maze-tutorial"}},[t._v("Maze tutorial")])]),a("li",[a("a",{attrs:{href:"#gymnasium-environment-with-scikit-decide-tutorial-continuous-mountain-car"}},[t._v("Gymnasium environment with scikit-decide tutorial: Continuous Mountain Car")])]),a("li",[a("a",{attrs:{href:"#introduction-to-scheduling"}},[t._v("Introduction to scheduling")])]),a("li",[a("a",{attrs:{href:"#benchmarking-scikit-decide-solvers"}},[t._v("Benchmarking scikit-decide solvers")])]),a("li",[a("a",{attrs:{href:"#flight-planning-domain"}},[t._v("Flight Planning Domain")])])])]),a("p"),t._v(" "),a("h2",{attrs:{id:"maze-tutorial"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#maze-tutorial"}},[t._v("#")]),t._v(" Maze tutorial")]),t._v(" "),a("p",[a("a",{attrs:{href:"https://github.com/nhuet/scikit-decide/blob/notebooks-v1.0.0/notebooks/11_maze_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://img.shields.io/badge/see-Github-579aca?logo=github",alt:"Github"}}),a("OutboundLink")],1),t._v(" "),a("a",{attrs:{href:"https://colab.research.google.com/github/nhuet/scikit-decide/blob/notebooks-v1.0.0/notebooks/11_maze_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Colab"}}),a("OutboundLink")],1),t._v(" "),a("a",{attrs:{href:"https://mybinder.org/v2/gh/nhuet/scikit-decide/notebooks-v1.0.0?labpath=notebooks%2F11_maze_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://mybinder.org/badge_logo.svg",alt:"Binder"}}),a("OutboundLink")],1)]),t._v(" "),a("p",[t._v("In this tutorial, we tackle the maze problem.\nWe use this classical game to demonstrate how")]),t._v(" "),a("ul",[a("li",[t._v("a new scikit-decide domain can be easily created")]),t._v(" "),a("li",[t._v("to find solvers from scikit-decide hub matching its characteristics")]),t._v(" "),a("li",[t._v("to apply a scikit-decide solver to a domain")]),t._v(" "),a("li",[t._v("to create its own rollout function to play a trained solver on a domain")])]),t._v(" "),a("p",[t._v("Notes:")]),t._v(" "),a("ul",[a("li",[t._v("In order to focus on scikit-decide use, we put some code not directly related to the library in a "),a("a",{attrs:{href:"./maze_utils.py"}},[t._v("separate module")]),t._v(" (like maze generation and display).")]),t._v(" "),a("li",[t._v("A similar maze domain is already defined in "),a("a",{attrs:{href:"https://github.com/airbus/scikit-decide/blob/master/skdecide/hub/domain/maze/maze.py",target:"_blank",rel:"noopener noreferrer"}},[t._v("scikit-decide hub"),a("OutboundLink")],1),t._v(" but we do not use it for the sake of this tutorial.")]),t._v(" "),a("li",[a("strong",[t._v("Special notice for binder + sb3:")]),t._v("\nit seems that "),a("a",{attrs:{href:"https://stable-baselines3.readthedocs.io/en/master/",target:"_blank",rel:"noopener noreferrer"}},[t._v("stable-baselines3"),a("OutboundLink")],1),t._v(" algorithms are extremely slow on "),a("a",{attrs:{href:"https://mybinder.org/",target:"_blank",rel:"noopener noreferrer"}},[t._v("binder"),a("OutboundLink")],1),t._v(". We could not find a proper explanation about it. We strongly advise you to either launch the notebook locally or on colab, or to skip the cells that are using sb3 algorithms (here PPO solver).")])]),t._v(" "),a("h2",{attrs:{id:"gymnasium-environment-with-scikit-decide-tutorial-continuous-mountain-car"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#gymnasium-environment-with-scikit-decide-tutorial-continuous-mountain-car"}},[t._v("#")]),t._v(" Gymnasium environment with scikit-decide tutorial: Continuous Mountain Car")]),t._v(" "),a("p",[a("a",{attrs:{href:"https://github.com/nhuet/scikit-decide/blob/notebooks-v1.0.0/notebooks/12_gym_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://img.shields.io/badge/see-Github-579aca?logo=github",alt:"Github"}}),a("OutboundLink")],1),t._v(" "),a("a",{attrs:{href:"https://colab.research.google.com/github/nhuet/scikit-decide/blob/notebooks-v1.0.0/notebooks/12_gym_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Colab"}}),a("OutboundLink")],1),t._v(" "),a("a",{attrs:{href:"https://mybinder.org/v2/gh/nhuet/scikit-decide/notebooks-v1.0.0?labpath=notebooks%2F12_gym_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://mybinder.org/badge_logo.svg",alt:"Binder"}}),a("OutboundLink")],1)]),t._v(" "),a("p",[t._v("In this notebook we tackle the continuous mountain car problem taken from "),a("a",{attrs:{href:"https://gymnasium.farama.org/",target:"_blank",rel:"noopener noreferrer"}},[t._v("Gymnasium"),a("OutboundLink")],1),t._v(" (previously OpenAI Gym), a toolkit for developing environments, usually to be solved by Reinforcement Learning (RL) algorithms.")]),t._v(" "),a("p",[t._v("Continuous Mountain Car, a standard testing domain in RL, is a problem in which an under-powered car must drive up a steep hill.")]),t._v(" "),a("div",{attrs:{align:"middle"}},[a("img",{staticStyle:{width:"200px"},attrs:{alt:"mountain_car_continuous.gif",src:"https://gymnasium.farama.org/_images/mountain_car_continuous.gif"}})]),t._v(" "),a("p",[t._v("Note that we use here the "),a("em",[t._v("continuous")]),t._v(" version of the mountain car because\nit has a "),a("em",[t._v("shaped")]),t._v(" or "),a("em",[t._v("dense")]),t._v(' reward (i.e. not sparse) which can be used successfully when solving, as opposed to the other "Mountain Car" environments.\nFor reminder, a sparse reward is a reward which is null almost everywhere, whereas a dense or shaped reward has more meaningful values for most transitions.')]),t._v(" "),a("p",[t._v("This problem has been chosen for two reasons:")]),t._v(" "),a("ul",[a("li",[t._v("Show how scikit-decide can be used to solve gymnasium environments (the de-facto standard in the RL community),")]),t._v(" "),a("li",[t._v("Highlight that by doing so, you will be able to use not only solvers from the RL community (like the ones in "),a("a",{attrs:{href:"https://github.com/DLR-RM/stable-baselines3",target:"_blank",rel:"noopener noreferrer"}},[t._v("stable_baselines3"),a("OutboundLink")],1),t._v(" for example), but also other solvers coming from other communities like genetic programming and planning/search (use of an underlying search graph) that can be very efficient.")])]),t._v(" "),a("p",[t._v("Therefore in this notebook we will go through the following steps:")]),t._v(" "),a("ul",[a("li",[t._v("Wrap a gymnasium environment in a scikit-decide domain;")]),t._v(" "),a("li",[t._v("Use a classical RL algorithm like PPO to solve our problem;")]),t._v(" "),a("li",[t._v("Give CGP (Cartesian Genetic Programming)  a try on the same problem;")]),t._v(" "),a("li",[t._v("Finally use IW (Iterated Width) coming from the planning community on the same problem.")])]),t._v(" "),a("p",[a("strong",[t._v("Special notice for binder + sb3:")]),t._v("\nit seems that "),a("a",{attrs:{href:"https://stable-baselines3.readthedocs.io/en/master/",target:"_blank",rel:"noopener noreferrer"}},[t._v("stable-baselines3"),a("OutboundLink")],1),t._v(" algorithms are extremely slow on "),a("a",{attrs:{href:"https://mybinder.org/",target:"_blank",rel:"noopener noreferrer"}},[t._v("binder"),a("OutboundLink")],1),t._v(". We could not find a proper explanation about it. We strongly advise you to either launch the notebook locally or on colab, or to skip the cells that are using sb3 algorithms (here PPO solver).")]),t._v(" "),a("h2",{attrs:{id:"introduction-to-scheduling"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#introduction-to-scheduling"}},[t._v("#")]),t._v(" Introduction to scheduling")]),t._v(" "),a("p",[a("a",{attrs:{href:"https://github.com/nhuet/scikit-decide/blob/notebooks-v1.0.0/notebooks/13_scheduling_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://img.shields.io/badge/see-Github-579aca?logo=github",alt:"Github"}}),a("OutboundLink")],1),t._v(" "),a("a",{attrs:{href:"https://colab.research.google.com/github/nhuet/scikit-decide/blob/notebooks-v1.0.0/notebooks/13_scheduling_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Colab"}}),a("OutboundLink")],1),t._v(" "),a("a",{attrs:{href:"https://mybinder.org/v2/gh/nhuet/scikit-decide/notebooks-v1.0.0?labpath=notebooks%2F13_scheduling_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://mybinder.org/badge_logo.svg",alt:"Binder"}}),a("OutboundLink")],1)]),t._v(" "),a("p",[t._v("In this notebook, we explore how to solve a resource constrained project scheduling problem (RCPSP).")]),t._v(" "),a("p",[t._v("The problem is made of "),a("mjx-container",{staticClass:"MathJax",attrs:{jax:"SVG"}},[a("svg",{staticStyle:{"vertical-align":"0"},attrs:{xmlns:"http://www.w3.org/2000/svg",width:"2.378ex",height:"1.545ex",viewBox:"0 -683 1051 683"}},[a("g",{attrs:{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"matrix(1 0 0 -1 0 0)"}},[a("g",{attrs:{"data-mml-node":"math"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"4D",d:"M289 629Q289 635 232 637Q208 637 201 638T194 648Q194 649 196 659Q197 662 198 666T199 671T201 676T203 679T207 681T212 683T220 683T232 684Q238 684 262 684T307 683Q386 683 398 683T414 678Q415 674 451 396L487 117L510 154Q534 190 574 254T662 394Q837 673 839 675Q840 676 842 678T846 681L852 683H948Q965 683 988 683T1017 684Q1051 684 1051 673Q1051 668 1048 656T1045 643Q1041 637 1008 637Q968 636 957 634T939 623Q936 618 867 340T797 59Q797 55 798 54T805 50T822 48T855 46H886Q892 37 892 35Q892 19 885 5Q880 0 869 0Q864 0 828 1T736 2Q675 2 644 2T609 1Q592 1 592 11Q592 13 594 25Q598 41 602 43T625 46Q652 46 685 49Q699 52 704 61Q706 65 742 207T813 490T848 631L654 322Q458 10 453 5Q451 4 449 3Q444 0 433 0Q418 0 415 7Q413 11 374 317L335 624L267 354Q200 88 200 79Q206 46 272 46H282Q288 41 289 37T286 19Q282 3 278 1Q274 0 267 0Q265 0 255 0T221 1T157 2Q127 2 95 1T58 0Q43 0 39 2T35 11Q35 13 38 25T43 40Q45 46 65 46Q135 46 154 86Q158 92 223 354T289 629Z"}})])])])])]),t._v(" activities that have precedence constraints. That means that if activity "),a("mjx-container",{staticClass:"MathJax",attrs:{jax:"SVG"}},[a("svg",{staticStyle:{"vertical-align":"-0.566ex"},attrs:{xmlns:"http://www.w3.org/2000/svg",width:"9.471ex",height:"2.262ex",viewBox:"0 -750 4186.2 1000"}},[a("g",{attrs:{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"matrix(1 0 0 -1 0 0)"}},[a("g",{attrs:{"data-mml-node":"math"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"6A",d:"M297 596Q297 627 318 644T361 661Q378 661 389 651T403 623Q403 595 384 576T340 557Q322 557 310 567T297 596ZM288 376Q288 405 262 405Q240 405 220 393T185 362T161 325T144 293L137 279Q135 278 121 278H107Q101 284 101 286T105 299Q126 348 164 391T252 441Q253 441 260 441T272 442Q296 441 316 432Q341 418 354 401T367 348V332L318 133Q267 -67 264 -75Q246 -125 194 -164T75 -204Q25 -204 7 -183T-12 -137Q-12 -110 7 -91T53 -71Q70 -71 82 -81T95 -112Q95 -148 63 -167Q69 -168 77 -168Q111 -168 139 -140T182 -74L193 -32Q204 11 219 72T251 197T278 308T289 365Q289 372 288 376Z"}})]),a("g",{attrs:{"data-mml-node":"mo",transform:"translate(689.8, 0)"}},[a("path",{attrs:{"data-c":"2208",d:"M84 250Q84 372 166 450T360 539Q361 539 377 539T419 540T469 540H568Q583 532 583 520Q583 511 570 501L466 500Q355 499 329 494Q280 482 242 458T183 409T147 354T129 306T124 272V270H568Q583 262 583 250T568 230H124V228Q124 207 134 177T167 112T231 48T328 7Q355 1 466 0H570Q583 -10 583 -20Q583 -32 568 -40H471Q464 -40 446 -40T417 -41Q262 -41 172 45Q84 127 84 250Z"}})]),a("g",{attrs:{"data-mml-node":"mo",transform:"translate(1634.6, 0)"}},[a("path",{attrs:{"data-c":"5B",d:"M118 -250V750H255V710H158V-210H255V-250H118Z"}})]),a("g",{attrs:{"data-mml-node":"mn",transform:"translate(1912.6, 0)"}},[a("path",{attrs:{"data-c":"31",d:"M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"}})]),a("g",{attrs:{"data-mml-node":"mo",transform:"translate(2412.6, 0)"}},[a("path",{attrs:{"data-c":"2C",d:"M78 35T78 60T94 103T137 121Q165 121 187 96T210 8Q210 -27 201 -60T180 -117T154 -158T130 -185T117 -194Q113 -194 104 -185T95 -172Q95 -168 106 -156T131 -126T157 -76T173 -3V9L172 8Q170 7 167 6T161 3T152 1T140 0Q113 0 96 17Z"}})]),a("g",{attrs:{"data-mml-node":"mi",transform:"translate(2857.2, 0)"}},[a("path",{attrs:{"data-c":"4D",d:"M289 629Q289 635 232 637Q208 637 201 638T194 648Q194 649 196 659Q197 662 198 666T199 671T201 676T203 679T207 681T212 683T220 683T232 684Q238 684 262 684T307 683Q386 683 398 683T414 678Q415 674 451 396L487 117L510 154Q534 190 574 254T662 394Q837 673 839 675Q840 676 842 678T846 681L852 683H948Q965 683 988 683T1017 684Q1051 684 1051 673Q1051 668 1048 656T1045 643Q1041 637 1008 637Q968 636 957 634T939 623Q936 618 867 340T797 59Q797 55 798 54T805 50T822 48T855 46H886Q892 37 892 35Q892 19 885 5Q880 0 869 0Q864 0 828 1T736 2Q675 2 644 2T609 1Q592 1 592 11Q592 13 594 25Q598 41 602 43T625 46Q652 46 685 49Q699 52 704 61Q706 65 742 207T813 490T848 631L654 322Q458 10 453 5Q451 4 449 3Q444 0 433 0Q418 0 415 7Q413 11 374 317L335 624L267 354Q200 88 200 79Q206 46 272 46H282Q288 41 289 37T286 19Q282 3 278 1Q274 0 267 0Q265 0 255 0T221 1T157 2Q127 2 95 1T58 0Q43 0 39 2T35 11Q35 13 38 25T43 40Q45 46 65 46Q135 46 154 86Q158 92 223 354T289 629Z"}})]),a("g",{attrs:{"data-mml-node":"mo",transform:"translate(3908.2, 0)"}},[a("path",{attrs:{"data-c":"5D",d:"M22 710V750H159V-250H22V-210H119V710H22Z"}})])])])])]),t._v(" is a successor of activity "),a("mjx-container",{staticClass:"MathJax",attrs:{jax:"SVG"}},[a("svg",{staticStyle:{"vertical-align":"-0.566ex"},attrs:{xmlns:"http://www.w3.org/2000/svg",width:"9.32ex",height:"2.262ex",viewBox:"0 -750 4119.2 1000"}},[a("g",{attrs:{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"matrix(1 0 0 -1 0 0)"}},[a("g",{attrs:{"data-mml-node":"math"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"69",d:"M184 600Q184 624 203 642T247 661Q265 661 277 649T290 619Q290 596 270 577T226 557Q211 557 198 567T184 600ZM21 287Q21 295 30 318T54 369T98 420T158 442Q197 442 223 419T250 357Q250 340 236 301T196 196T154 83Q149 61 149 51Q149 26 166 26Q175 26 185 29T208 43T235 78T260 137Q263 149 265 151T282 153Q302 153 302 143Q302 135 293 112T268 61T223 11T161 -11Q129 -11 102 10T74 74Q74 91 79 106T122 220Q160 321 166 341T173 380Q173 404 156 404H154Q124 404 99 371T61 287Q60 286 59 284T58 281T56 279T53 278T49 278T41 278H27Q21 284 21 287Z"}})]),a("g",{attrs:{"data-mml-node":"mo",transform:"translate(622.8, 0)"}},[a("path",{attrs:{"data-c":"2208",d:"M84 250Q84 372 166 450T360 539Q361 539 377 539T419 540T469 540H568Q583 532 583 520Q583 511 570 501L466 500Q355 499 329 494Q280 482 242 458T183 409T147 354T129 306T124 272V270H568Q583 262 583 250T568 230H124V228Q124 207 134 177T167 112T231 48T328 7Q355 1 466 0H570Q583 -10 583 -20Q583 -32 568 -40H471Q464 -40 446 -40T417 -41Q262 -41 172 45Q84 127 84 250Z"}})]),a("g",{attrs:{"data-mml-node":"mo",transform:"translate(1567.6, 0)"}},[a("path",{attrs:{"data-c":"5B",d:"M118 -250V750H255V710H158V-210H255V-250H118Z"}})]),a("g",{attrs:{"data-mml-node":"mn",transform:"translate(1845.6, 0)"}},[a("path",{attrs:{"data-c":"31",d:"M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"}})]),a("g",{attrs:{"data-mml-node":"mo",transform:"translate(2345.6, 0)"}},[a("path",{attrs:{"data-c":"2C",d:"M78 35T78 60T94 103T137 121Q165 121 187 96T210 8Q210 -27 201 -60T180 -117T154 -158T130 -185T117 -194Q113 -194 104 -185T95 -172Q95 -168 106 -156T131 -126T157 -76T173 -3V9L172 8Q170 7 167 6T161 3T152 1T140 0Q113 0 96 17Z"}})]),a("g",{attrs:{"data-mml-node":"mi",transform:"translate(2790.2, 0)"}},[a("path",{attrs:{"data-c":"4D",d:"M289 629Q289 635 232 637Q208 637 201 638T194 648Q194 649 196 659Q197 662 198 666T199 671T201 676T203 679T207 681T212 683T220 683T232 684Q238 684 262 684T307 683Q386 683 398 683T414 678Q415 674 451 396L487 117L510 154Q534 190 574 254T662 394Q837 673 839 675Q840 676 842 678T846 681L852 683H948Q965 683 988 683T1017 684Q1051 684 1051 673Q1051 668 1048 656T1045 643Q1041 637 1008 637Q968 636 957 634T939 623Q936 618 867 340T797 59Q797 55 798 54T805 50T822 48T855 46H886Q892 37 892 35Q892 19 885 5Q880 0 869 0Q864 0 828 1T736 2Q675 2 644 2T609 1Q592 1 592 11Q592 13 594 25Q598 41 602 43T625 46Q652 46 685 49Q699 52 704 61Q706 65 742 207T813 490T848 631L654 322Q458 10 453 5Q451 4 449 3Q444 0 433 0Q418 0 415 7Q413 11 374 317L335 624L267 354Q200 88 200 79Q206 46 272 46H282Q288 41 289 37T286 19Q282 3 278 1Q274 0 267 0Q265 0 255 0T221 1T157 2Q127 2 95 1T58 0Q43 0 39 2T35 11Q35 13 38 25T43 40Q45 46 65 46Q135 46 154 86Q158 92 223 354T289 629Z"}})]),a("g",{attrs:{"data-mml-node":"mo",transform:"translate(3841.2, 0)"}},[a("path",{attrs:{"data-c":"5D",d:"M22 710V750H159V-250H22V-210H119V710H22Z"}})])])])])]),t._v(", then activity "),a("mjx-container",{staticClass:"MathJax",attrs:{jax:"SVG"}},[a("svg",{staticStyle:{"vertical-align":"-0.025ex"},attrs:{xmlns:"http://www.w3.org/2000/svg",width:"0.781ex",height:"1.52ex",viewBox:"0 -661 345 672"}},[a("g",{attrs:{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"matrix(1 0 0 -1 0 0)"}},[a("g",{attrs:{"data-mml-node":"math"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"69",d:"M184 600Q184 624 203 642T247 661Q265 661 277 649T290 619Q290 596 270 577T226 557Q211 557 198 567T184 600ZM21 287Q21 295 30 318T54 369T98 420T158 442Q197 442 223 419T250 357Q250 340 236 301T196 196T154 83Q149 61 149 51Q149 26 166 26Q175 26 185 29T208 43T235 78T260 137Q263 149 265 151T282 153Q302 153 302 143Q302 135 293 112T268 61T223 11T161 -11Q129 -11 102 10T74 74Q74 91 79 106T122 220Q160 321 166 341T173 380Q173 404 156 404H154Q124 404 99 371T61 287Q60 286 59 284T58 281T56 279T53 278T49 278T41 278H27Q21 284 21 287Z"}})])])])])]),t._v(" must be completed before activity "),a("mjx-container",{staticClass:"MathJax",attrs:{jax:"SVG"}},[a("svg",{staticStyle:{"vertical-align":"-0.462ex"},attrs:{xmlns:"http://www.w3.org/2000/svg",width:"0.932ex",height:"1.957ex",viewBox:"0 -661 412 865"}},[a("g",{attrs:{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"matrix(1 0 0 -1 0 0)"}},[a("g",{attrs:{"data-mml-node":"math"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"6A",d:"M297 596Q297 627 318 644T361 661Q378 661 389 651T403 623Q403 595 384 576T340 557Q322 557 310 567T297 596ZM288 376Q288 405 262 405Q240 405 220 393T185 362T161 325T144 293L137 279Q135 278 121 278H107Q101 284 101 286T105 299Q126 348 164 391T252 441Q253 441 260 441T272 442Q296 441 316 432Q341 418 354 401T367 348V332L318 133Q267 -67 264 -75Q246 -125 194 -164T75 -204Q25 -204 7 -183T-12 -137Q-12 -110 7 -91T53 -71Q70 -71 82 -81T95 -112Q95 -148 63 -167Q69 -168 77 -168Q111 -168 139 -140T182 -74L193 -32Q204 11 219 72T251 197T278 308T289 365Q289 372 288 376Z"}})])])])])]),t._v(" can be started")],1),t._v(" "),a("p",[t._v("On top of these constraints, each project is assigned a set of K renewable resources where each resource "),a("mjx-container",{staticClass:"MathJax",attrs:{jax:"SVG"}},[a("svg",{staticStyle:{"vertical-align":"-0.025ex"},attrs:{xmlns:"http://www.w3.org/2000/svg",width:"1.179ex",height:"1.595ex",viewBox:"0 -694 521 705"}},[a("g",{attrs:{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"matrix(1 0 0 -1 0 0)"}},[a("g",{attrs:{"data-mml-node":"math"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"6B",d:"M121 647Q121 657 125 670T137 683Q138 683 209 688T282 694Q294 694 294 686Q294 679 244 477Q194 279 194 272Q213 282 223 291Q247 309 292 354T362 415Q402 442 438 442Q468 442 485 423T503 369Q503 344 496 327T477 302T456 291T438 288Q418 288 406 299T394 328Q394 353 410 369T442 390L458 393Q446 405 434 405H430Q398 402 367 380T294 316T228 255Q230 254 243 252T267 246T293 238T320 224T342 206T359 180T365 147Q365 130 360 106T354 66Q354 26 381 26Q429 26 459 145Q461 153 479 153H483Q499 153 499 144Q499 139 496 130Q455 -11 378 -11Q333 -11 305 15T277 90Q277 108 280 121T283 145Q283 167 269 183T234 206T200 217T182 220H180Q168 178 159 139T145 81T136 44T129 20T122 7T111 -2Q98 -11 83 -11Q66 -11 57 -1T48 16Q48 26 85 176T158 471L195 616Q196 629 188 632T149 637H144Q134 637 131 637T124 640T121 647Z"}})])])])])]),t._v(" is available in "),a("mjx-container",{staticClass:"MathJax",attrs:{jax:"SVG"}},[a("svg",{staticStyle:{"vertical-align":"-0.357ex"},attrs:{xmlns:"http://www.w3.org/2000/svg",width:"2.664ex",height:"1.902ex",viewBox:"0 -683 1177.4 840.8"}},[a("g",{attrs:{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"matrix(1 0 0 -1 0 0)"}},[a("g",{attrs:{"data-mml-node":"math"}},[a("g",{attrs:{"data-mml-node":"msub"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"52",d:"M230 637Q203 637 198 638T193 649Q193 676 204 682Q206 683 378 683Q550 682 564 680Q620 672 658 652T712 606T733 563T739 529Q739 484 710 445T643 385T576 351T538 338L545 333Q612 295 612 223Q612 212 607 162T602 80V71Q602 53 603 43T614 25T640 16Q668 16 686 38T712 85Q717 99 720 102T735 105Q755 105 755 93Q755 75 731 36Q693 -21 641 -21H632Q571 -21 531 4T487 82Q487 109 502 166T517 239Q517 290 474 313Q459 320 449 321T378 323H309L277 193Q244 61 244 59Q244 55 245 54T252 50T269 48T302 46H333Q339 38 339 37T336 19Q332 6 326 0H311Q275 2 180 2Q146 2 117 2T71 2T50 1Q33 1 33 10Q33 12 36 24Q41 43 46 45Q50 46 61 46H67Q94 46 127 49Q141 52 146 61Q149 65 218 339T287 628Q287 635 230 637ZM630 554Q630 586 609 608T523 636Q521 636 500 636T462 637H440Q393 637 386 627Q385 624 352 494T319 361Q319 360 388 360Q466 361 492 367Q556 377 592 426Q608 449 619 486T630 554Z"}})]),a("g",{attrs:{"data-mml-node":"TeXAtom",transform:"translate(759, -150) scale(0.707)"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"6B",d:"M121 647Q121 657 125 670T137 683Q138 683 209 688T282 694Q294 694 294 686Q294 679 244 477Q194 279 194 272Q213 282 223 291Q247 309 292 354T362 415Q402 442 438 442Q468 442 485 423T503 369Q503 344 496 327T477 302T456 291T438 288Q418 288 406 299T394 328Q394 353 410 369T442 390L458 393Q446 405 434 405H430Q398 402 367 380T294 316T228 255Q230 254 243 252T267 246T293 238T320 224T342 206T359 180T365 147Q365 130 360 106T354 66Q354 26 381 26Q429 26 459 145Q461 153 479 153H483Q499 153 499 144Q499 139 496 130Q455 -11 378 -11Q333 -11 305 15T277 90Q277 108 280 121T283 145Q283 167 269 183T234 206T200 217T182 220H180Q168 178 159 139T145 81T136 44T129 20T122 7T111 -2Q98 -11 83 -11Q66 -11 57 -1T48 16Q48 26 85 176T158 471L195 616Q196 629 188 632T149 637H144Q134 637 131 637T124 640T121 647Z"}})])])])])])])]),t._v(" units for the entire duration of the project. Each activity may require one or more of these resources to be completed. While scheduling the activities, the daily resource usage for resource "),a("mjx-container",{staticClass:"MathJax",attrs:{jax:"SVG"}},[a("svg",{staticStyle:{"vertical-align":"-0.025ex"},attrs:{xmlns:"http://www.w3.org/2000/svg",width:"1.179ex",height:"1.595ex",viewBox:"0 -694 521 705"}},[a("g",{attrs:{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"matrix(1 0 0 -1 0 0)"}},[a("g",{attrs:{"data-mml-node":"math"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"6B",d:"M121 647Q121 657 125 670T137 683Q138 683 209 688T282 694Q294 694 294 686Q294 679 244 477Q194 279 194 272Q213 282 223 291Q247 309 292 354T362 415Q402 442 438 442Q468 442 485 423T503 369Q503 344 496 327T477 302T456 291T438 288Q418 288 406 299T394 328Q394 353 410 369T442 390L458 393Q446 405 434 405H430Q398 402 367 380T294 316T228 255Q230 254 243 252T267 246T293 238T320 224T342 206T359 180T365 147Q365 130 360 106T354 66Q354 26 381 26Q429 26 459 145Q461 153 479 153H483Q499 153 499 144Q499 139 496 130Q455 -11 378 -11Q333 -11 305 15T277 90Q277 108 280 121T283 145Q283 167 269 183T234 206T200 217T182 220H180Q168 178 159 139T145 81T136 44T129 20T122 7T111 -2Q98 -11 83 -11Q66 -11 57 -1T48 16Q48 26 85 176T158 471L195 616Q196 629 188 632T149 637H144Q134 637 131 637T124 640T121 647Z"}})])])])])]),t._v(" can not exceed "),a("mjx-container",{staticClass:"MathJax",attrs:{jax:"SVG"}},[a("svg",{staticStyle:{"vertical-align":"-0.357ex"},attrs:{xmlns:"http://www.w3.org/2000/svg",width:"2.664ex",height:"1.902ex",viewBox:"0 -683 1177.4 840.8"}},[a("g",{attrs:{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"matrix(1 0 0 -1 0 0)"}},[a("g",{attrs:{"data-mml-node":"math"}},[a("g",{attrs:{"data-mml-node":"msub"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"52",d:"M230 637Q203 637 198 638T193 649Q193 676 204 682Q206 683 378 683Q550 682 564 680Q620 672 658 652T712 606T733 563T739 529Q739 484 710 445T643 385T576 351T538 338L545 333Q612 295 612 223Q612 212 607 162T602 80V71Q602 53 603 43T614 25T640 16Q668 16 686 38T712 85Q717 99 720 102T735 105Q755 105 755 93Q755 75 731 36Q693 -21 641 -21H632Q571 -21 531 4T487 82Q487 109 502 166T517 239Q517 290 474 313Q459 320 449 321T378 323H309L277 193Q244 61 244 59Q244 55 245 54T252 50T269 48T302 46H333Q339 38 339 37T336 19Q332 6 326 0H311Q275 2 180 2Q146 2 117 2T71 2T50 1Q33 1 33 10Q33 12 36 24Q41 43 46 45Q50 46 61 46H67Q94 46 127 49Q141 52 146 61Q149 65 218 339T287 628Q287 635 230 637ZM630 554Q630 586 609 608T523 636Q521 636 500 636T462 637H440Q393 637 386 627Q385 624 352 494T319 361Q319 360 388 360Q466 361 492 367Q556 377 592 426Q608 449 619 486T630 554Z"}})]),a("g",{attrs:{"data-mml-node":"TeXAtom",transform:"translate(759, -150) scale(0.707)"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"6B",d:"M121 647Q121 657 125 670T137 683Q138 683 209 688T282 694Q294 694 294 686Q294 679 244 477Q194 279 194 272Q213 282 223 291Q247 309 292 354T362 415Q402 442 438 442Q468 442 485 423T503 369Q503 344 496 327T477 302T456 291T438 288Q418 288 406 299T394 328Q394 353 410 369T442 390L458 393Q446 405 434 405H430Q398 402 367 380T294 316T228 255Q230 254 243 252T267 246T293 238T320 224T342 206T359 180T365 147Q365 130 360 106T354 66Q354 26 381 26Q429 26 459 145Q461 153 479 153H483Q499 153 499 144Q499 139 496 130Q455 -11 378 -11Q333 -11 305 15T277 90Q277 108 280 121T283 145Q283 167 269 183T234 206T200 217T182 220H180Q168 178 159 139T145 81T136 44T129 20T122 7T111 -2Q98 -11 83 -11Q66 -11 57 -1T48 16Q48 26 85 176T158 471L195 616Q196 629 188 632T149 637H144Q134 637 131 637T124 640T121 647Z"}})])])])])])])]),t._v(" units.")],1),t._v(" "),a("p",[t._v("Each activity "),a("mjx-container",{staticClass:"MathJax",attrs:{jax:"SVG"}},[a("svg",{staticStyle:{"vertical-align":"-0.462ex"},attrs:{xmlns:"http://www.w3.org/2000/svg",width:"0.932ex",height:"1.957ex",viewBox:"0 -661 412 865"}},[a("g",{attrs:{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"matrix(1 0 0 -1 0 0)"}},[a("g",{attrs:{"data-mml-node":"math"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"6A",d:"M297 596Q297 627 318 644T361 661Q378 661 389 651T403 623Q403 595 384 576T340 557Q322 557 310 567T297 596ZM288 376Q288 405 262 405Q240 405 220 393T185 362T161 325T144 293L137 279Q135 278 121 278H107Q101 284 101 286T105 299Q126 348 164 391T252 441Q253 441 260 441T272 442Q296 441 316 432Q341 418 354 401T367 348V332L318 133Q267 -67 264 -75Q246 -125 194 -164T75 -204Q25 -204 7 -183T-12 -137Q-12 -110 7 -91T53 -71Q70 -71 82 -81T95 -112Q95 -148 63 -167Q69 -168 77 -168Q111 -168 139 -140T182 -74L193 -32Q204 11 219 72T251 197T278 308T289 365Q289 372 288 376Z"}})])])])])]),t._v(" takes "),a("mjx-container",{staticClass:"MathJax",attrs:{jax:"SVG"}},[a("svg",{staticStyle:{"vertical-align":"-0.666ex"},attrs:{xmlns:"http://www.w3.org/2000/svg",width:"1.949ex",height:"2.236ex",viewBox:"0 -694 861.3 988.2"}},[a("g",{attrs:{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"matrix(1 0 0 -1 0 0)"}},[a("g",{attrs:{"data-mml-node":"math"}},[a("g",{attrs:{"data-mml-node":"msub"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"64",d:"M366 683Q367 683 438 688T511 694Q523 694 523 686Q523 679 450 384T375 83T374 68Q374 26 402 26Q411 27 422 35Q443 55 463 131Q469 151 473 152Q475 153 483 153H487H491Q506 153 506 145Q506 140 503 129Q490 79 473 48T445 8T417 -8Q409 -10 393 -10Q359 -10 336 5T306 36L300 51Q299 52 296 50Q294 48 292 46Q233 -10 172 -10Q117 -10 75 30T33 157Q33 205 53 255T101 341Q148 398 195 420T280 442Q336 442 364 400Q369 394 369 396Q370 400 396 505T424 616Q424 629 417 632T378 637H357Q351 643 351 645T353 664Q358 683 366 683ZM352 326Q329 405 277 405Q242 405 210 374T160 293Q131 214 119 129Q119 126 119 118T118 106Q118 61 136 44T179 26Q233 26 290 98L298 109L352 326Z"}})]),a("g",{attrs:{"data-mml-node":"TeXAtom",transform:"translate(520, -150) scale(0.707)"}},[a("g",{attrs:{"data-mml-node":"mi"}},[a("path",{attrs:{"data-c":"6A",d:"M297 596Q297 627 318 644T361 661Q378 661 389 651T403 623Q403 595 384 576T340 557Q322 557 310 567T297 596ZM288 376Q288 405 262 405Q240 405 220 393T185 362T161 325T144 293L137 279Q135 278 121 278H107Q101 284 101 286T105 299Q126 348 164 391T252 441Q253 441 260 441T272 442Q296 441 316 432Q341 418 354 401T367 348V332L318 133Q267 -67 264 -75Q246 -125 194 -164T75 -204Q25 -204 7 -183T-12 -137Q-12 -110 7 -91T53 -71Q70 -71 82 -81T95 -112Q95 -148 63 -167Q69 -168 77 -168Q111 -168 139 -140T182 -74L193 -32Q204 11 219 72T251 197T278 308T289 365Q289 372 288 376Z"}})])])])])])])]),t._v(" time units to complete.")],1),t._v(" "),a("p",[t._v("The overall goal of the problem is usually to minimize the makespan.")]),t._v(" "),a("p",[t._v("A classic variant of RCPSP is the multimode RCPSP where each task can be executed in several ways (one way=one mode). A typical example is :")]),t._v(" "),a("ul",[a("li",[t._v("Mode n°1 'Fast mode': high resource consumption and fast")]),t._v(" "),a("li",[t._v("Mode n°2 'Slow mode' : low resource consumption but slow")])]),t._v(" "),a("h2",{attrs:{id:"benchmarking-scikit-decide-solvers"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#benchmarking-scikit-decide-solvers"}},[t._v("#")]),t._v(" Benchmarking scikit-decide solvers")]),t._v(" "),a("p",[a("a",{attrs:{href:"https://github.com/nhuet/scikit-decide/blob/notebooks-v1.0.0/notebooks/14_benchmarking_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://img.shields.io/badge/see-Github-579aca?logo=github",alt:"Github"}}),a("OutboundLink")],1),t._v(" "),a("a",{attrs:{href:"https://colab.research.google.com/github/nhuet/scikit-decide/blob/notebooks-v1.0.0/notebooks/14_benchmarking_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Colab"}}),a("OutboundLink")],1),t._v(" "),a("a",{attrs:{href:"https://mybinder.org/v2/gh/nhuet/scikit-decide/notebooks-v1.0.0?labpath=notebooks%2F14_benchmarking_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://mybinder.org/badge_logo.svg",alt:"Binder"}}),a("OutboundLink")],1)]),t._v(" "),a("p",[t._v("This notebook demonstrates how to run and compare scikit-decide solvers compatible with a given domain.")]),t._v(" "),a("p",[t._v("This benchmark is supported by "),a("a",{attrs:{href:"https://docs.ray.io/en/latest/tune/index.html",target:"_blank",rel:"noopener noreferrer"}},[t._v("Ray Tune"),a("OutboundLink")],1),t._v(", a scalable Python library for experiment execution and hyperparameter tuning (incl. running experiments in parallel and logging results to Tensorboard).")]),t._v(" "),a("p",[t._v("Benchmarking is important since the most efficient solvers might greatly vary depending on the domain.")]),t._v(" "),a("h2",{attrs:{id:"flight-planning-domain"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#flight-planning-domain"}},[t._v("#")]),t._v(" Flight Planning Domain")]),t._v(" "),a("p",[a("a",{attrs:{href:"https://github.com/nhuet/scikit-decide/blob/notebooks-v1.0.0/notebooks/15_flightplanning_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://img.shields.io/badge/see-Github-579aca?logo=github",alt:"Github"}}),a("OutboundLink")],1),t._v(" "),a("a",{attrs:{href:"https://colab.research.google.com/github/nhuet/scikit-decide/blob/notebooks-v1.0.0/notebooks/15_flightplanning_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Colab"}}),a("OutboundLink")],1),t._v(" "),a("a",{attrs:{href:"https://mybinder.org/v2/gh/nhuet/scikit-decide/notebooks-v1.0.0?labpath=notebooks%2F15_flightplanning_tuto.ipynb",target:"_blank",rel:"noopener noreferrer"}},[a("img",{attrs:{src:"https://mybinder.org/badge_logo.svg",alt:"Binder"}}),a("OutboundLink")],1)]),t._v(" "),a("p",[t._v("This notebook aims to make a short and interactive example of the Flight Planning Domain. See the "),a("a",{attrs:{href:"https://airbus.github.io/scikit-decide/reference/_skdecide.hub.domain.flight_planning.domain.html#flightplanningdomain",target:"_blank",rel:"noopener noreferrer"}},[t._v("online documentation"),a("OutboundLink")],1),t._v(" for more information.")])])}),[],!1,null,null,null);e.default=o.exports}}]);