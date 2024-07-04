(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[299],{89924:function(e,t,n){"use strict";n.d(t,{y:function(){return v}});var r=n(9669),o=n.n(r),i=n(84506),s=n(68253),a=n(35823),c=n.n(a),u=i,l=o().create({baseURL:u.url_root+"/api/v1/",headers:{"Access-Control-Allow-Origin":"*"}});l.interceptors.request.use((function(e){return e.headers.Authorization="Bearer "+(0,s.Gg)().user.token,e}),(function(e){console.log("Interceptor request error: "+e)})),l.interceptors.response.use((function(e){return e}),(function(e){throw console.log(" AXIOS error: "),console.log(e),401===e.response.status&&(0,s.KY)("","",-1,0,"",!0),403===e.response.status&&console.log("Error: "+e.response.data.status),404===e.response.status&&console.log("Error: "+e.response.data.status),409===e.response.status&&console.log("Error: "+e.response.data.status),422===e.response.status&&(0,s.KY)("","",-1,0,"",!0),e}));var d=function(e){return e.data},f=function(e){return l.get(e).then(d)},p=function(e,t,n){return l.post(e,{pin:n},{responseType:"blob"}).then((function(e){t=e.headers["content-disposition"].split('"')[1],c()(e.data,t)}))},h=function(e,t){return l.post(e,t).then(d)},m=function(e,t){return l.patch(e,t).then(d)},g=function(e){return l.delete(e).then(d)},v={login:function(e){return h("login",e)},getUsers:function(){return f("users")},getUser:function(e){return f("users/".concat(e))},getUserStartupKit:function(e,t,n){return p("users/".concat(e,"/blob"),t,n)},getClientStartupKit:function(e,t,n){return p("clients/".concat(e,"/blob"),t,n)},getOverseerStartupKit:function(e,t){return p("overseer/blob",e,t)},getServerStartupKit:function(e,t,n){return p("servers/".concat(e,"/blob"),t,n)},getClients:function(){return f("clients")},getProject:function(){return f("project")},postUser:function(e){return h("users",e)},patchUser:function(e,t){return m("users/".concat(e),t)},deleteUser:function(e){return g("users/".concat(e))},postClient:function(e){return h("clients",e)},patchClient:function(e,t){return m("clients/".concat(e),t)},deleteClient:function(e){return g("clients/".concat(e))},patchProject:function(e){return m("project",e)},getServer:function(){return f("server")},patchServer:function(e){return m("server",e)}}},57061:function(e,t,n){"use strict";n.d(t,{Z:function(){return P}});var r=n(50029),o=n(87794),i=n.n(o),s=n(41664),a=n.n(s),c=n(29224),u=n.n(c),l=n(70491),d=n.n(l),f=n(86188),p=n.n(f),h=n(29430),m=h.default.div.withConfig({displayName:"styles__StyledLayout",componentId:"sc-xczy9u-0"})(["overflow:hidden;height:100%;width:100%;margin:0;padding:0;display:flex;flex-wrap:wrap;.menu{height:auto;}.content-header{flex:0 0 80px;}"]),g=h.default.div.withConfig({displayName:"styles__StyledContent",componentId:"sc-xczy9u-1"})(["display:flex;flex-direction:column;flex:1 1 0%;overflow:auto;height:calc(100% - 3rem);.inlineeditlarger{padding:10px;}.inlineedit{padding:10px;margin:-10px;}.content-wrapper{padding:",";min-height:800px;}"],(function(e){return e.theme.spacing.four})),v=n(11163),b=n(84506),j=n(67294),x=n(68253),y=n(13258),w=n.n(y),S=n(5801),O=n.n(S),C=n(89924),_=n(85893),P=function(e){var t=e.children,n=e.headerChildren,o=e.title,s=(0,v.useRouter)(),c=s.pathname,l=s.push,h=b,S=(0,j.useState)(),P=S[0],N=S[1],I=(0,x.Gg)();(0,j.useEffect)((function(){C.y.getProject().then((function(e){N(e.project)}))}),[]);var k=function(){var e=(0,r.Z)(i().mark((function e(){return i().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:(0,x.KY)("none","",-1,0),l("/");case 2:case"end":return e.stop()}}),e)})));return function(){return e.apply(this,arguments)}}();return(0,_.jsxs)(m,{children:[(0,_.jsx)(u(),{app:null===P||void 0===P?void 0:P.short_name,appBarActions:I.user.role>0?(0,_.jsxs)("div",{style:{display:"flex",flexDirection:"row",alignItems:"center",marginRight:10},children:[h.demo&&(0,_.jsx)("div",{children:"DEMO MODE"}),(0,_.jsx)(w(),{parentElement:(0,_.jsx)(O(),{icon:{name:"AccountCircleGenericUser",color:"white",size:22},shape:"circle",variant:"link",className:"logout-link"}),position:"top-right",children:(0,_.jsxs)(_.Fragment,{children:[(0,_.jsx)(y.ActionMenuItem,{label:"Logout",onClick:k}),!1]})})]}):(0,_.jsx)(_.Fragment,{})}),(0,_.jsxs)(p(),{className:"menu",itemMatchPattern:function(e){return e===c},itemRenderer:function(e){var t=e.title,n=e.href;return(0,_.jsx)(a(),{href:n,children:t})},location:c,children:[0==I.user.role&&(0,_.jsxs)(f.MenuContent,{children:[(0,_.jsx)(f.MenuItem,{href:"/",icon:{name:"AccountUser"},title:"Login"}),(0,_.jsx)(f.MenuItem,{href:"/registration-form",icon:{name:"ObjectsClipboardEdit"},title:"User Registration Form"})]}),4==I.user.role&&(0,_.jsxs)(f.MenuContent,{children:[(0,_.jsx)(f.MenuItem,{href:"/",icon:{name:"ViewList"},title:"Project Home"}),(0,_.jsx)(f.MenuItem,{href:"/user-dashboard",icon:{name:"ServerEdit"},title:"My Info"}),(0,_.jsx)(f.MenuItem,{href:"/project-admin-dashboard",icon:{name:"AccountGroupShieldAdd"},title:"Users Dashboard"}),(0,_.jsx)(f.MenuItem,{href:"/site-dashboard",icon:{name:"ConnectionNetworkComputers2"},title:"Client Sites"}),(0,_.jsx)(f.MenuItem,{href:"/project-configuration",icon:{name:"SettingsCog"},title:"Project Configuration"}),(0,_.jsx)(f.MenuItem,{href:"/server-config",icon:{name:"ConnectionServerNetwork1"},title:"Server Configuration"}),(0,_.jsx)(f.MenuItem,{href:"/downloads",icon:{name:"ActionsDownload"},title:"Downloads"}),(0,_.jsx)(f.MenuItem,{href:"/logout",icon:{name:"PlaybackStop"},title:"Logout"})]}),(1==I.user.role||2==I.user.role||3==I.user.role)&&(0,_.jsxs)(f.MenuContent,{children:[(0,_.jsx)(f.MenuItem,{href:"/",icon:{name:"ViewList"},title:"Project Home Page"}),(0,_.jsx)(f.MenuItem,{href:"/user-dashboard",icon:{name:"ServerEdit"},title:"My Info"}),(0,_.jsx)(f.MenuItem,{href:"/downloads",icon:{name:"ActionsDownload"},title:"Downloads"}),(0,_.jsx)(f.MenuItem,{href:"/logout",icon:{name:"PlaybackStop"},title:"Logout"})]}),(0,_.jsx)(f.MenuFooter,{})]}),(0,_.jsxs)(g,{children:[(0,_.jsx)(d(),{className:"content-header",title:o,children:n}),(0,_.jsx)("div",{className:"content-wrapper",children:t})]})]})}},4636:function(e,t,n){"use strict";n.d(t,{mg:function(){return j},nv:function(){return C}});n(67294);var r=n(82175),o=n(29430),i=(n(40398),n(90878)),s=n.n(i),a=((0,o.default)(s()).withConfig({displayName:"ErrorMessage",componentId:"sc-azomh6-0"})(["display:block;color:",";font-size:",";font-weight:",";position:absolute;"],(function(e){return e.theme.colors.red500}),(function(e){return e.theme.typography.size.small}),(function(e){return e.theme.typography.weight.bold})),n(85893));o.default.div.withConfig({displayName:"CheckboxField__CheckboxWrapper",componentId:"sc-1m8bzxk-0"})(["align-self:",";display:",";.checkbox-label-wrapper{display:flex;align-items:flex-start;.checkbox-custom-label{margin-left:",";}}.checkbox-text,.checkbox-custom-label{font-size:",";white-space:break-spaces;}"],(function(e){return e.centerVertically?"center":""}),(function(e){return e.centerVertically&&"flex"}),(function(e){return e.theme.spacing.one}),(function(e){return e.theme.typography.size.small}));var c=n(59499),u=n(4730),l=n(32754),d=n.n(l),f=n(57299),p=o.default.div.withConfig({displayName:"InfoMessage__FieldHint",componentId:"sc-s0s5lu-0"})(["display:flex;align-items:center;svg{margin-right:",";}"],(function(e){return e.theme.spacing.one})),h=function(e){var t=e.children;return(0,a.jsxs)(p,{children:[(0,a.jsx)(f.default,{name:"StatusCircleInformation",size:"small"}),t]})},m=o.default.div.withConfig({displayName:"styles__StyledField",componentId:"sc-1fjauag-0"})(["> div{margin-bottom:0px;}margin-bottom:",";"],(function(e){return e.theme.spacing.four})),g=["name","disabled","options","placeholder","info","wrapperClass"];function v(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function b(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?v(Object(n),!0).forEach((function(t){(0,c.Z)(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):v(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}var j=function(e){var t=e.name,n=e.disabled,o=e.options,i=e.placeholder,s=e.info,c=e.wrapperClass,l=void 0===c?"":c,f=(0,u.Z)(e,g);return(0,a.jsx)(r.gN,{name:t,children:function(e){var r=e.form,c=!!r.errors[t]&&(r.submitCount>0||r.touched[t]);return(0,a.jsxs)(m,{className:l,children:[(0,a.jsx)(d(),b(b({},f),{},{disabled:!!n||r.isSubmitting,name:t,onBlur:function(){return r.setFieldTouched(t,!0)},onChange:function(e){return r.setFieldValue(t,e)},options:o,placeholder:i,valid:!c,validationMessage:c?r.errors[t]:void 0,value:null===r||void 0===r?void 0:r.values[t]})),s&&(0,a.jsx)(h,{children:s})]})}})},x=n(24777),y=n.n(x),w=["className","name","info","label","disabled","placeholder"];function S(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function O(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?S(Object(n),!0).forEach((function(t){(0,c.Z)(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):S(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}var C=function(e){var t=e.className,n=e.name,o=e.info,i=e.label,s=e.disabled,c=e.placeholder,l=(0,u.Z)(e,w);return(0,a.jsx)(r.gN,{name:n,children:function(e){var r=e.form,u=!!r.errors[n]&&(r.submitCount>0||r.touched[n]);return(0,a.jsxs)(m,{children:[(0,a.jsx)(y(),O(O({},l),{},{className:t,disabled:!!s||r.isSubmitting,label:i,name:n,onBlur:r.handleBlur,onChange:r.handleChange,placeholder:c,valid:!u,validationMessage:u?r.errors[n]:void 0,value:null===r||void 0===r?void 0:r.values[n]})),o&&(0,a.jsx)(h,{children:o})]})}})}},56921:function(e,t,n){"use strict";n.d(t,{P:function(){return u},X:function(){return c}});var r=n(29430),o=n(36578),i=n.n(o),s=n(3159),a=n.n(s),c=(0,r.default)(a()).withConfig({displayName:"form-page__StyledFormExample",componentId:"sc-rfrcq8-0"})([".bottom{display:flex;gap:",";}.zero-left{margin-left:0;}.zero-right{margin-right:0;}"],(function(e){return e.theme.spacing.four})),u=(0,r.default)(i()).withConfig({displayName:"form-page__StyledBanner",componentId:"sc-rfrcq8-1"})(["margin-bottom:1rem;"])},44060:function(e,t,n){"use strict";n.r(t);var r=n(82175),o=n(74231),i=n(5801),s=n.n(i),a=n(40398),c=n.n(a),u=n(56921),l=n(57061),d=n(4636),f=n(67294),p=n(89924),h=n(11163),m=n(85893),g=o.Ry().shape({server1:o.Z_().required("Required"),overseer:o.Z_(),server2:o.Z_()});t.default=function(){var e=(0,h.useRouter)().push,t=(0,f.useState)(),n=t[0],o=t[1];return(0,f.useEffect)((function(){p.y.getProject().then((function(e){o(e.project)}))}),[]),(0,m.jsx)(m.Fragment,{children:(0,m.jsx)(l.Z,{title:"Server Configuration:",children:n?(0,m.jsx)(r.J9,{initialValues:n,onSubmit:function(t,n){p.y.patchProject({server1:t.server1,ha_mode:t.ha_mode,overseer:t.overseer,server2:t.server2}).then((function(e){"ok"==e.status&&o(e.project),setTimeout((function(){n.setSubmitting(!1),n.resetForm({values:e.project})}),500)})).catch((function(t){console.log(t),e("/")}))},validationSchema:g,children:function(e){return(0,m.jsxs)(m.Fragment,{children:[e.values.frozen&&(0,m.jsx)(u.P,{status:"warning",rounded:!0,children:"This project has been frozen. Project values are no longer editable."}),(0,m.jsxs)(u.X,{loading:e.isSubmitting,title:"",children:[(0,m.jsx)(d.nv,{disabled:e.isSubmitting||e.values.frozen,label:"Server (DNS name)",name:"server1",placeholder:""}),(0,m.jsx)(r.gN,{as:c(),disabled:e.isSubmitting||e.values.frozen,label:"Enable HA (High Availability)",name:"ha_mode",checked:e.values.ha_mode}),e.values.ha_mode?(0,m.jsxs)("div",{className:"bottom",children:[(0,m.jsx)(d.nv,{disabled:e.isSubmitting||e.values.frozen,label:"Overseer (DNS name)",name:"overseer",placeholder:""}),(0,m.jsx)(d.nv,{disabled:e.isSubmitting||e.values.frozen,label:"Optional Backup Server (DNS name)",name:"server2",placeholder:""})]}):(0,m.jsx)("div",{children:(0,m.jsx)("br",{})}),(0,m.jsx)(s(),{disabled:!e.dirty||!e.isValid||e.values.frozen,onClick:function(){return e.handleSubmit()},children:"Save"})]})]})}}):(0,m.jsx)("span",{children:"loading..."})})})}},68253:function(e,t,n){"use strict";n.d(t,{Gg:function(){return s},KY:function(){return o},a5:function(){return i}});var r={user:{email:"",token:"",id:-1,role:0},status:"unauthenticated"};function o(e,t,n,o,i){var s=arguments.length>5&&void 0!==arguments[5]&&arguments[5];return r={user:{email:e,token:t,id:n,role:o,org:i},expired:s,status:0==o?"unauthenticated":"authenticated"},localStorage.setItem("session",JSON.stringify(r)),r}function i(e){return r={user:{email:r.user.email,token:r.user.token,id:r.user.id,role:e,org:r.user.org},expired:r.expired,status:r.status},localStorage.setItem("session",JSON.stringify(r)),r}function s(){var e=localStorage.getItem("session");return null!=e&&(r=JSON.parse(e)),r}},24043:function(e,t,n){(window.__NEXT_P=window.__NEXT_P||[]).push(["/server-config",function(){return n(44060)}])},50029:function(e,t,n){"use strict";function r(e,t,n,r,o,i,s){try{var a=e[i](s),c=a.value}catch(u){return void n(u)}a.done?t(c):Promise.resolve(c).then(r,o)}function o(e){return function(){var t=this,n=arguments;return new Promise((function(o,i){var s=e.apply(t,n);function a(e){r(s,o,i,a,c,"next",e)}function c(e){r(s,o,i,a,c,"throw",e)}a(void 0)}))}}n.d(t,{Z:function(){return o}})},84506:function(e){"use strict";e.exports=JSON.parse('{"projectname":"New FL Project","demo":false,"url_root":"","arraydata":[{"name":"itemone"},{"name":"itemtwo"},{"name":"itemthree"}]}')}},function(e){e.O(0,[234,897,774,888,179],(function(){return t=24043,e(e.s=t);var t}));var t=e.O();_N_E=t}]);