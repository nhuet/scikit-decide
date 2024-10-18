(window.webpackJsonp=window.webpackJsonp||[]).push([[76],{589:function(t,a,e){"use strict";e.r(a);var n=e(38),r=Object(n.a)({},(function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[e("h1",{attrs:{id:"hub-domain-flight-planning-weather-interpolator-weather-tools-interpolator-weatherinterpolator"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#hub-domain-flight-planning-weather-interpolator-weather-tools-interpolator-weatherinterpolator"}},[t._v("#")]),t._v(" hub.domain.flight_planning.weather_interpolator.weather_tools.interpolator.WeatherInterpolator")]),t._v(" "),e("div",{staticClass:"custom-block tip"},[e("p",{staticClass:"custom-block-title"},[t._v("Domain specification")]),t._v(" "),e("skdecide-summary")],1),t._v(" "),e("h2",{attrs:{id:"weatherforecastinterpolator"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#weatherforecastinterpolator"}},[t._v("#")]),t._v(" WeatherForecastInterpolator")]),t._v(" "),e("p",[t._v("Class used to store weather data, interpolate and plot weather forecast from .npz files")]),t._v(" "),e("h3",{attrs:{id:"constructor"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#constructor"}},[t._v("#")]),t._v(" Constructor "),e("Badge",{attrs:{text:"WeatherForecastInterpolator",type:"tip"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"WeatherForecastInterpolator",sig:{params:[{name:"file_npz"},{name:"time_cut_index",default:"None"},{name:"order_interp",default:"1"}]}}}),t._v(" "),e("p",[t._v("Stores the weather data and build the interpolators on grid.")]),t._v(" "),e("h3",{attrs:{id:"interpol-field"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#interpol-field"}},[t._v("#")]),t._v(" interpol_field "),e("Badge",{attrs:{text:"WeatherInterpolator",type:"warn"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"interpol_field",sig:{params:[{name:"self"},{name:"X"},{name:"field",default:"temperature"}]}}}),t._v(" "),e("p",[t._v("Interpol one field that is present in interpolators for array of 4d points")]),t._v(" "),e("p",[t._v(":param X: array of points [time (in s), alt (in ft), lat, long]\n:param field: field of weather data to interpolate (could be 'temperature' or 'humidity'\n:return: array of interpolated values")]),t._v(" "),e("h3",{attrs:{id:"interpol-wind"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#interpol-wind"}},[t._v("#")]),t._v(" interpol_wind "),e("Badge",{attrs:{text:"WeatherInterpolator",type:"warn"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"interpol_wind",sig:{params:[{name:"self"},{name:"X"}]}}}),t._v(" "),e("p",[t._v("Interpol wind for an array of 4D points")]),t._v(" "),e("p",[t._v(":param X: array of points [time (in s), alt (in ft), lat, long]\n:return: wind vector.")]),t._v(" "),e("h3",{attrs:{id:"interpol-wind-classic"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#interpol-wind-classic"}},[t._v("#")]),t._v(" interpol_wind_classic "),e("Badge",{attrs:{text:"WeatherInterpolator",type:"warn"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"interpol_wind_classic",sig:{params:[{name:"self"},{name:"lat"},{name:"longi"},{name:"alt",default:"35000.0"},{name:"t",default:"0.0"},{name:"**kwargs"}]}}}),t._v(" "),e("p",[t._v("Interpol the wind in one 4D point")]),t._v(" "),e("p",[t._v(":param lat: latitude\n:param longi: longitude\n:param alt: altitude (in ft)\n:param t: time in second\n:return: [wind strength, direction, wind wector]")]),t._v(" "),e("h3",{attrs:{id:"plot-field"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#plot-field"}},[t._v("#")]),t._v(" plot_field "),e("Badge",{attrs:{text:"WeatherForecastInterpolator",type:"tip"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"plot_field",sig:{params:[{name:"self"},{name:"field",default:"issr"},{name:"alt",default:"35000.0"},{name:"t",default:"0.0"},{name:"n_lat",default:"180"},{name:"n_long",default:"720"},{name:"ax",default:"None"}]}}}),t._v(" "),e("p",[t._v("Plot a field for a given altitude and for one time/or range of time")]),t._v(" "),e("p",[t._v(":param alt: altitude couch to plot (in ft)\n:param t: value of time step (in second) or list/array of time step (in s)\n:param n_lat: number of latitude discretized steps\n:param n_long: number of longitude discretized steps\n:param ax: Ax object where to plot the wind (possibily a precomputed basemap or classic ax object)")]),t._v(" "),e("h3",{attrs:{id:"plot-matrix-wind"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#plot-matrix-wind"}},[t._v("#")]),t._v(" plot_matrix_wind "),e("Badge",{attrs:{text:"WeatherForecastInterpolator",type:"tip"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"plot_matrix_wind",sig:{params:[{name:"self"},{name:"index_alt",default:"10"},{name:"index_time",default:"0"},{name:"ax",default:"None"}]}}}),t._v(" "),e("p",[t._v("[Deprecated]")]),t._v(" "),e("p",[t._v("Plot the wind matrix directly (no interpolation contrary\nto :func:"),e("code",[t._v("BEN3_G.Contrails.WeatherForecastInterpolator.plot_wind")])]),t._v(" "),e("h3",{attrs:{id:"plot-matrix-wind-noised"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#plot-matrix-wind-noised"}},[t._v("#")]),t._v(" plot_matrix_wind_noised "),e("Badge",{attrs:{text:"WeatherForecastInterpolator",type:"tip"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"plot_matrix_wind_noised",sig:{params:[{name:"self"},{name:"index_alt",default:"10"},{name:"index_time",default:"0"},{name:"ax",default:"None"}]}}}),t._v(" "),e("p",[t._v("[Deprecated]")]),t._v(" "),e("p",[t._v("Plot the wind matrix directly (no interpolation contrary\nto :func:"),e("code",[t._v("BEN3_G.Contrails.WeatherForecastInterpolator.plot_wind")])]),t._v(" "),e("h3",{attrs:{id:"plot-wind"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#plot-wind"}},[t._v("#")]),t._v(" plot_wind "),e("Badge",{attrs:{text:"WeatherForecastInterpolator",type:"tip"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"plot_wind",sig:{params:[{name:"self"},{name:"alt",default:"35000.0"},{name:"down_long",default:"-180.0"},{name:"up_long",default:"180.0"},{name:"down_lat",default:"-90.0"},{name:"up_lat",default:"90.0"},{name:"t",default:"0.0"},{name:"n_lat",default:"180"},{name:"n_long",default:"720"},{name:"plot_wind",default:"False"},{name:"ax",default:"None"}]}}}),t._v(" "),e("p",[t._v("Plot the wind for a given coordinates window for a given altitude and for one time/or range of time")]),t._v(" "),e("p",[t._v(":param alt: altitude couch to plot (in ft)\n:param down_long: min longitude\n:param up_long: max longitude\n:param down_lat: min latitude\n:param up_lat: max latitude\n:param t: value of time step (in second) or list/array of time step (in s)\n:type t: float or iterable\n:param n_lat: number of latitude discretized steps\n:param n_long: number of longitude discretized steps\n:param plot_wind: plot the vector field\n:param ax: Ax object where to plot the wind (possibily a precomputed basemap or classic ax object)")]),t._v(" "),e("h3",{attrs:{id:"plot-wind-noised"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#plot-wind-noised"}},[t._v("#")]),t._v(" plot_wind_noised "),e("Badge",{attrs:{text:"WeatherForecastInterpolator",type:"tip"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"plot_wind_noised",sig:{params:[{name:"self"},{name:"alt",default:"35000.0"},{name:"down_long",default:"-180.0"},{name:"up_long",default:"180.0"},{name:"down_lat",default:"-90.0"},{name:"up_lat",default:"90.0"},{name:"t",default:"0.0"},{name:"n_lat",default:"180"},{name:"n_long",default:"720"},{name:"plot_wind",default:"False"},{name:"mean_noised_norm",default:"1.05"},{name:"scale_noised_norm",default:"0.01"},{name:"mean_noised_arg",default:"0.1"},{name:"scale_noised_arg",default:"0.01"},{name:"ax",default:"None"}]}}}),t._v(" "),e("p",[t._v("Plot the wind for a given coordinates window for a given altitude and for one time/or range of time")]),t._v(" "),e("p",[t._v(":param alt: altitude couch to plot (in ft)\n:param down_long: min longitude\n:param up_long: max longitude\n:param down_lat: min latitude\n:param up_lat: max latitude\n:param t: value of time step (in second) or list/array of time step (in s)\n:type t: float or iterable\n:param n_lat: number of latitude discretized steps\n:param n_long: number of longitude discretized steps\n:param plot_wind: plot the vector field\n:param ax: Ax object where to plot the wind (possibily a precomputed basemap or classic ax object)")]),t._v(" "),e("h3",{attrs:{id:"transform-long"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#transform-long"}},[t._v("#")]),t._v(" transform_long "),e("Badge",{attrs:{text:"WeatherForecastInterpolator",type:"tip"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"transform_long",sig:{params:[{name:"self"},{name:"long"}]}}}),t._v(" "),e("p",[t._v("[Deprecated] should be replaced by modulo function...")]),t._v(" "),e("p",[t._v(":param long: array of longitudes\n:return: array of longitude put in positive domain (modulo 360.)")])],1)}),[],!1,null,null,null);a.default=r.exports}}]);