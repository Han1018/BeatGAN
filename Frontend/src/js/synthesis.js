!function(e,t){"object"==typeof exports&&"object"==typeof module?module.exports=t():"function"==typeof define&&define.amd?define([],t):"object"==typeof exports?exports.synthesisjs=t():e.synthesisjs=t()}(window,(function(){return function(e){var t={};function r(s){if(t[s])return t[s].exports;var n=t[s]={i:s,l:!1,exports:{}};return e[s].call(n.exports,n,n.exports,r),n.l=!0,n.exports}return r.m=e,r.c=t,r.d=function(e,t,s){r.o(e,t)||Object.defineProperty(e,t,{enumerable:!0,get:s})},r.r=function(e){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},r.t=function(e,t){if(1&t&&(e=r(e)),8&t)return e;if(4&t&&"object"==typeof e&&e&&e.__esModule)return e;var s=Object.create(null);if(r.r(s),Object.defineProperty(s,"default",{enumerable:!0,value:e}),2&t&&"string"!=typeof e)for(var n in e)r.d(s,n,function(t){return e[t]}.bind(null,n));return s},r.n=function(e){var t=e&&e.__esModule?function(){return e.default}:function(){return e};return r.d(t,"a",t),t},r.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},r.p="",r(r.s=1)}([function(e,t,r){"use strict";function s(e,t,r,n){this.message=e,this.expected=t,this.found=r,this.location=n,this.name="SyntaxError","function"==typeof Error.captureStackTrace&&Error.captureStackTrace(this,s)}!function(e,t){function r(){this.constructor=e}r.prototype=t.prototype,e.prototype=new r}(s,Error),s.buildMessage=function(e,t){var r={literal:function(e){return'"'+n(e.text)+'"'},class:function(e){var t,r="";for(t=0;t<e.parts.length;t++)r+=e.parts[t]instanceof Array?i(e.parts[t][0])+"-"+i(e.parts[t][1]):i(e.parts[t]);return"["+(e.inverted?"^":"")+r+"]"},any:function(e){return"any character"},end:function(e){return"end of input"},other:function(e){return e.description}};function s(e){return e.charCodeAt(0).toString(16).toUpperCase()}function n(e){return e.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\0/g,"\\0").replace(/\t/g,"\\t").replace(/\n/g,"\\n").replace(/\r/g,"\\r").replace(/[\x00-\x0F]/g,(function(e){return"\\x0"+s(e)})).replace(/[\x10-\x1F\x7F-\x9F]/g,(function(e){return"\\x"+s(e)}))}function i(e){return e.replace(/\\/g,"\\\\").replace(/\]/g,"\\]").replace(/\^/g,"\\^").replace(/-/g,"\\-").replace(/\0/g,"\\0").replace(/\t/g,"\\t").replace(/\n/g,"\\n").replace(/\r/g,"\\r").replace(/[\x00-\x0F]/g,(function(e){return"\\x0"+s(e)})).replace(/[\x10-\x1F\x7F-\x9F]/g,(function(e){return"\\x"+s(e)}))}return"Expected "+function(e){var t,s,n,i=new Array(e.length);for(t=0;t<e.length;t++)i[t]=(n=e[t],r[n.type](n));if(i.sort(),i.length>0){for(t=1,s=1;t<i.length;t++)i[t-1]!==i[t]&&(i[s]=i[t],s++);i.length=s}switch(i.length){case 1:return i[0];case 2:return i[0]+" or "+i[1];default:return i.slice(0,-1).join(", ")+", or "+i[i.length-1]}}(e)+" but "+function(e){return e?'"'+n(e)+'"':"end of input"}(t)+" found."},e.exports={SyntaxError:s,parse:function(e,t){t=void 0!==t?t:{};var r,n={},i={start:ne},a=ne,o=Y(";",!1),c=/^[ \t\r\n]/,h=Z([" ","\t","\r","\n"],!1,!1),l=Y("/*",!1),u=Y("*/",!1),f={type:"any"},d=Y("//",!1),p=/^[^\n]/,g=Z(["\n"],!0,!1),m=/^[cdefgab]/,v=Z(["c","d","e","f","g","a","b"],!1,!1),A=/^[\-+]/,b=Z(["-","+"],!1,!1),y=/^[0-9]/,C=Z([["0","9"]],!1,!1),k=Y(".",!1),x=Y("^",!1),w=(Y("&",!1),Y("r",!1)),P=Y("o",!1),M=Y("-",!1),S=Y("<",!1),B=Y(">",!1),O=Y("l",!1),T=Y("q",!1),_=/^[1-8]/,I=Z([["1","8"]],!1,!1),E=Y("u",!1),$=Y("v",!1),z=Y("p",!1),D=Y("E",!1),F=Y("B",!1),j=Y(",",!1),q=Y("@",!1),R=Y("D",!1),N=Y("t",!1),W=Y("?",!1),L=Y("k",!1),H=Y("C",!1),V=0,U=0,G=[{line:1,column:1}],J=0,K=[],Q=0;if("startRule"in t){if(!(t.startRule in i))throw new Error("Can't start parsing from rule \""+t.startRule+'".');a=i[t.startRule]}function X(e,t){throw function(e,t){return new s(e,null,null,t)}(e,t=void 0!==t?t:te(U,V))}function Y(e,t){return{type:"literal",text:e,ignoreCase:t}}function Z(e,t,r){return{type:"class",parts:e,inverted:t,ignoreCase:r}}function ee(t){var r,s=G[t];if(s)return s;for(r=t-1;!G[r];)r--;for(s={line:(s=G[r]).line,column:s.column};r<t;)10===e.charCodeAt(r)?(s.line++,s.column=1):s.column++,r++;return G[t]=s,s}function te(e,t){var r=ee(e),s=ee(t);return{start:{offset:e,line:r.line,column:r.column},end:{offset:t,line:s.line,column:s.column}}}function re(e){V<J||(V>J&&(J=V,K=[]),K.push(e))}function se(e,t,r){return new s(s.buildMessage(e,t),e,t,r)}function ne(){var e,t;if(e=[],(t=ie())!==n)for(;t!==n;)e.push(t),t=ie();else e=n;return e}function ie(){var t,r,s;for(t=V,r=[],s=oe();s!==n;)r.push(s),s=oe();return r!==n&&(s=function(){var t,r;t=V,ae()!==n?(59===e.charCodeAt(V)?(r=";",V++):(r=n,0===Q&&re(o)),r!==n&&ae()!==n?(U=t,t=null):(V=t,t=n)):(V=t,t=n);return t}())!==n?(U=t,t=r=r):(V=t,t=n),t}function ae(){var t,r;for(t=[],c.test(e.charAt(V))?(r=e.charAt(V),V++):(r=n,0===Q&&re(h));r!==n;)t.push(r),c.test(e.charAt(V))?(r=e.charAt(V),V++):(r=n,0===Q&&re(h));return t}function oe(){var t;return(t=function(){var t,r,s,i,a,o,c;if(t=V,ae()!==n)if("/*"===e.substr(V,2)?(r="/*",V+=2):(r=n,0===Q&&re(l)),r!==n){for(s=V,i=[],a=V,o=V,Q++,"*/"===e.substr(V,2)?(c="*/",V+=2):(c=n,0===Q&&re(u)),Q--,c===n?o=void 0:(V=o,o=n),o!==n?(e.length>V?(c=e.charAt(V),V++):(c=n,0===Q&&re(f)),c!==n?a=o=[o,c]:(V=a,a=n)):(V=a,a=n);a!==n;)i.push(a),a=V,o=V,Q++,"*/"===e.substr(V,2)?(c="*/",V+=2):(c=n,0===Q&&re(u)),Q--,c===n?o=void 0:(V=o,o=n),o!==n?(e.length>V?(c=e.charAt(V),V++):(c=n,0===Q&&re(f)),c!==n?a=o=[o,c]:(V=a,a=n)):(V=a,a=n);(s=i!==n?e.substring(s,V):i)!==n?("*/"===e.substr(V,2)?(i="*/",V+=2):(i=n,0===Q&&re(u)),i!==n&&(a=ae())!==n?(U=t,t={command:"comment"}):(V=t,t=n)):(V=t,t=n)}else V=t,t=n;else V=t,t=n;if(t===n)if(t=V,ae()!==n)if("//"===e.substr(V,2)?(r="//",V+=2):(r=n,0===Q&&re(d)),r!==n){for(s=[],p.test(e.charAt(V))?(i=e.charAt(V),V++):(i=n,0===Q&&re(g));i!==n;)s.push(i),p.test(e.charAt(V))?(i=e.charAt(V),V++):(i=n,0===Q&&re(g));s!==n&&(i=ae())!==n?(U=t,t={command:"comment"}):(V=t,t=n)}else V=t,t=n;else V=t,t=n;return t}())===n&&(t=function(){var t,r,s,i,a,o,c,h;if(t=V,ae()!==n)if(m.test(e.charAt(V))?(r=e.charAt(V),V++):(r=n,0===Q&&re(v)),r!==n)if(ae()!==n){for(s=[],A.test(e.charAt(V))?(i=e.charAt(V),V++):(i=n,0===Q&&re(b));i!==n;)s.push(i),A.test(e.charAt(V))?(i=e.charAt(V),V++):(i=n,0===Q&&re(b));if(s!==n)if((i=ae())!==n){for(a=V,o=[],y.test(e.charAt(V))?(c=e.charAt(V),V++):(c=n,0===Q&&re(C));c!==n;)o.push(c),y.test(e.charAt(V))?(c=e.charAt(V),V++):(c=n,0===Q&&re(C));if((a=o!==n?e.substring(a,V):o)!==n)if((o=ae())!==n){for(c=[],46===e.charCodeAt(V)?(h=".",V++):(h=n,0===Q&&re(k));h!==n;)c.push(h),46===e.charCodeAt(V)?(h=".",V++):(h=n,0===Q&&re(k));c!==n&&(h=ae())!==n?(U=t,t={command:"note",tone:r,accidentals:s,length:+a,dots:c}):(V=t,t=n)}else V=t,t=n;else V=t,t=n}else V=t,t=n;else V=t,t=n}else V=t,t=n;else V=t,t=n;else V=t,t=n;return t}())===n&&(t=function(){var t,r,s,i,a,o;if(t=V,ae()!==n)if(94===e.charCodeAt(V)?(r="^",V++):(r=n,0===Q&&re(x)),r!==n)if(ae()!==n){for(s=V,i=[],y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C));a!==n;)i.push(a),y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C));if((s=i!==n?e.substring(s,V):i)!==n)if((i=ae())!==n){for(a=[],46===e.charCodeAt(V)?(o=".",V++):(o=n,0===Q&&re(k));o!==n;)a.push(o),46===e.charCodeAt(V)?(o=".",V++):(o=n,0===Q&&re(k));a!==n&&(o=ae())!==n?(U=t,t={command:"tie",length:+s,dots:a}):(V=t,t=n)}else V=t,t=n;else V=t,t=n}else V=t,t=n;else V=t,t=n;else V=t,t=n;return t}())===n&&(t=function(){var t,r,s,i,a,o;if(t=V,ae()!==n)if(114===e.charCodeAt(V)?(r="r",V++):(r=n,0===Q&&re(w)),r!==n)if(ae()!==n){for(s=V,i=[],y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C));a!==n;)i.push(a),y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C));if((s=i!==n?e.substring(s,V):i)!==n)if((i=ae())!==n){for(a=[],46===e.charCodeAt(V)?(o=".",V++):(o=n,0===Q&&re(k));o!==n;)a.push(o),46===e.charCodeAt(V)?(o=".",V++):(o=n,0===Q&&re(k));a!==n&&(o=ae())!==n?(U=t,t={command:"rest",length:+s,dots:a}):(V=t,t=n)}else V=t,t=n;else V=t,t=n}else V=t,t=n;else V=t,t=n;else V=t,t=n;return t}())===n&&(t=function(){var t,r,s,i,a,o,c;if(t=V,ae()!==n)if(111===e.charCodeAt(V)?(r="o",V++):(r=n,0===Q&&re(P)),r!==n)if(ae()!==n){if(s=V,i=V,45===e.charCodeAt(V)?(a="-",V++):(a=n,0===Q&&re(M)),a===n&&(a=null),a!==n){if(o=[],y.test(e.charAt(V))?(c=e.charAt(V),V++):(c=n,0===Q&&re(C)),c!==n)for(;c!==n;)o.push(c),y.test(e.charAt(V))?(c=e.charAt(V),V++):(c=n,0===Q&&re(C));else o=n;o!==n?i=a=[a,o]:(V=i,i=n)}else V=i,i=n;(s=i!==n?e.substring(s,V):i)!==n&&(i=ae())!==n?(U=t,((h=s)<-1||h>10)&&X("octave number is out of range"),t={command:"octave",number:+h}):(V=t,t=n)}else V=t,t=n;else V=t,t=n;else V=t,t=n;var h;return t}())===n&&(t=function(){var t,r;t=V,ae()!==n?(60===e.charCodeAt(V)?(r="<",V++):(r=n,0===Q&&re(S)),r!==n&&ae()!==n?(U=t,t={command:"octave_up"}):(V=t,t=n)):(V=t,t=n);return t}())===n&&(t=function(){var t,r;t=V,ae()!==n?(62===e.charCodeAt(V)?(r=">",V++):(r=n,0===Q&&re(B)),r!==n&&ae()!==n?(U=t,t={command:"octave_down"}):(V=t,t=n)):(V=t,t=n);return t}())===n&&(t=function(){var t,r,s,i,a,o;if(t=V,ae()!==n)if(108===e.charCodeAt(V)?(r="l",V++):(r=n,0===Q&&re(O)),r!==n)if(ae()!==n){if(s=V,i=[],y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C)),a!==n)for(;a!==n;)i.push(a),y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C));else i=n;if((s=i!==n?e.substring(s,V):i)!==n)if((i=ae())!==n){for(a=[],46===e.charCodeAt(V)?(o=".",V++):(o=n,0===Q&&re(k));o!==n;)a.push(o),46===e.charCodeAt(V)?(o=".",V++):(o=n,0===Q&&re(k));a!==n&&(o=ae())!==n?(U=t,t={command:"note_length",length:+s,dots:a}):(V=t,t=n)}else V=t,t=n;else V=t,t=n}else V=t,t=n;else V=t,t=n;else V=t,t=n;return t}())===n&&(t=function(){var t,r,s;t=V,ae()!==n?(113===e.charCodeAt(V)?(r="q",V++):(r=n,0===Q&&re(T)),r!==n&&ae()!==n?(_.test(e.charAt(V))?(s=e.charAt(V),V++):(s=n,0===Q&&re(I)),s!==n&&ae()!==n?(U=t,t={command:"gate_time",quantity:+s}):(V=t,t=n)):(V=t,t=n)):(V=t,t=n);return t}())===n&&(t=function(){var t,r,s,i,a;if(t=V,ae()!==n)if(117===e.charCodeAt(V)?(r="u",V++):(r=n,0===Q&&re(E)),r!==n)if(ae()!==n){if(s=V,i=[],y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C)),a!==n)for(;a!==n;)i.push(a),y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C));else i=n;(s=i!==n?e.substring(s,V):i)!==n&&(i=ae())!==n?(U=t,((o=+(o=s))<0||o>127)&&X("velocity is out of range (0-127)"),t={command:"velocity",value:o}):(V=t,t=n)}else V=t,t=n;else V=t,t=n;else V=t,t=n;var o;return t}())===n&&(t=function(){var t,r,s,i,a;if(t=V,ae()!==n)if(118===e.charCodeAt(V)?(r="v",V++):(r=n,0===Q&&re($)),r!==n)if(ae()!==n){if(s=V,i=[],y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C)),a!==n)for(;a!==n;)i.push(a),y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C));else i=n;(s=i!==n?e.substring(s,V):i)!==n&&(i=ae())!==n?(U=t,((o=+(o=s))<0||o>127)&&X("volume is out of range (0-127)"),t={command:"volume",value:o}):(V=t,t=n)}else V=t,t=n;else V=t,t=n;else V=t,t=n;var o;return t}())===n&&(t=function(){var t,r,s,i,a,o,c;if(t=V,ae()!==n)if(112===e.charCodeAt(V)?(r="p",V++):(r=n,0===Q&&re(z)),r!==n)if(ae()!==n){if(s=V,i=V,45===e.charCodeAt(V)?(a="-",V++):(a=n,0===Q&&re(M)),a===n&&(a=null),a!==n){if(o=[],y.test(e.charAt(V))?(c=e.charAt(V),V++):(c=n,0===Q&&re(C)),c!==n)for(;c!==n;)o.push(c),y.test(e.charAt(V))?(c=e.charAt(V),V++):(c=n,0===Q&&re(C));else o=n;o!==n?i=a=[a,o]:(V=i,i=n)}else V=i,i=n;(s=i!==n?e.substring(s,V):i)!==n&&(i=ae())!==n?(U=t,((h=+(h=s))<-64||h>63)&&X("pan is out of range (-64-63)"),t={command:"pan",value:h}):(V=t,t=n)}else V=t,t=n;else V=t,t=n;else V=t,t=n;var h;return t}())===n&&(t=function(){var t,r,s,i,a;if(t=V,ae()!==n)if(69===e.charCodeAt(V)?(r="E",V++):(r=n,0===Q&&re(D)),r!==n)if(ae()!==n){if(s=V,i=[],y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C)),a!==n)for(;a!==n;)i.push(a),y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C));else i=n;(s=i!==n?e.substring(s,V):i)!==n&&(i=ae())!==n?(U=t,((o=+(o=s))<0||o>127)&&X("expression is out of range (0-127)"),t={command:"expression",value:o}):(V=t,t=n)}else V=t,t=n;else V=t,t=n;else V=t,t=n;var o;return t}())===n&&(t=function(){var t,r,s,i,a,o,c,h;if(t=V,ae()!==n)if(66===e.charCodeAt(V)?(r="B",V++):(r=n,0===Q&&re(F)),r!==n)if(ae()!==n){if(s=V,i=[],y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C)),a!==n)for(;a!==n;)i.push(a),y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C));else i=n;if((s=i!==n?e.substring(s,V):i)!==n)if((i=ae())!==n)if(44===e.charCodeAt(V)?(a=",",V++):(a=n,0===Q&&re(j)),a!==n)if(ae()!==n){if(o=V,c=[],y.test(e.charAt(V))?(h=e.charAt(V),V++):(h=n,0===Q&&re(C)),h!==n)for(;h!==n;)c.push(h),y.test(e.charAt(V))?(h=e.charAt(V),V++):(h=n,0===Q&&re(C));else c=n;(o=c!==n?e.substring(o,V):c)!==n&&(c=ae())!==n?(U=t,u=o,((l=s)<0||l>119)&&X("control number is out of range (0-127)"),(u<0||u>127)&&X("control value is out of range (0-127)"),t={command:"control_change",number:l,value:u}):(V=t,t=n)}else V=t,t=n;else V=t,t=n;else V=t,t=n;else V=t,t=n}else V=t,t=n;else V=t,t=n;else V=t,t=n;var l,u;return t}())===n&&(t=function(){var t,r,s,i,a;if(t=V,ae()!==n)if(64===e.charCodeAt(V)?(r="@",V++):(r=n,0===Q&&re(q)),r!==n)if(ae()!==n){if(s=V,i=[],y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C)),a!==n)for(;a!==n;)i.push(a),y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C));else i=n;(s=i!==n?e.substring(s,V):i)!==n&&(i=ae())!==n?(U=t,((o=+(o=s))<0||o>127)&&X("program number is out of range (0-127)"),t={command:"program_change",number:o}):(V=t,t=n)}else V=t,t=n;else V=t,t=n;else V=t,t=n;var o;return t}())===n&&(t=function(){var t,r,s,i,a;if(t=V,ae()!==n)if(68===e.charCodeAt(V)?(r="D",V++):(r=n,0===Q&&re(R)),r!==n)if(ae()!==n){if(s=V,i=[],y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C)),a!==n)for(;a!==n;)i.push(a),y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C));else i=n;(s=i!==n?e.substring(s,V):i)!==n&&(i=ae())!==n?(U=t,((o=+(o=s))<0||o>127)&&X("channel aftertouch is out of range (0-127)"),t={command:"channel_aftertouch",value:o}):(V=t,t=n)}else V=t,t=n;else V=t,t=n;else V=t,t=n;var o;return t}())===n&&(t=function(){var t,r,s,i,a;if(t=V,ae()!==n)if(116===e.charCodeAt(V)?(r="t",V++):(r=n,0===Q&&re(N)),r!==n)if(ae()!==n){if(s=V,i=[],y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C)),a!==n)for(;a!==n;)i.push(a),y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C));else i=n;(s=i!==n?e.substring(s,V):i)!==n&&(i=ae())!==n?(U=t,t={command:"tempo",value:+s}):(V=t,t=n)}else V=t,t=n;else V=t,t=n;else V=t,t=n;return t}())===n&&(t=function(){var t,r;t=V,ae()!==n?(63===e.charCodeAt(V)?(r="?",V++):(r=n,0===Q&&re(W)),r!==n&&ae()!==n?(U=t,t={command:"start_point"}):(V=t,t=n)):(V=t,t=n);return t}())===n&&(t=function(){var t,r,s,i,a,o,c;if(t=V,ae()!==n)if(107===e.charCodeAt(V)?(r="k",V++):(r=n,0===Q&&re(L)),r!==n)if(ae()!==n){if(s=V,i=V,45===e.charCodeAt(V)?(a="-",V++):(a=n,0===Q&&re(M)),a===n&&(a=null),a!==n){if(o=[],y.test(e.charAt(V))?(c=e.charAt(V),V++):(c=n,0===Q&&re(C)),c!==n)for(;c!==n;)o.push(c),y.test(e.charAt(V))?(c=e.charAt(V),V++):(c=n,0===Q&&re(C));else o=n;o!==n?i=a=[a,o]:(V=i,i=n)}else V=i,i=n;(s=i!==n?e.substring(s,V):i)!==n&&(i=ae())!==n?(U=t,((h=+(h=s))<-127||h>127)&&X("key shift is out of range (-127-127)"),t={command:"key_shift",value:h}):(V=t,t=n)}else V=t,t=n;else V=t,t=n;else V=t,t=n;var h;return t}())===n&&(t=function(){var t,r,s,i,a;if(t=V,ae()!==n)if(67===e.charCodeAt(V)?(r="C",V++):(r=n,0===Q&&re(H)),r!==n)if(ae()!==n){if(s=V,i=[],y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C)),a!==n)for(;a!==n;)i.push(a),y.test(e.charAt(V))?(a=e.charAt(V),V++):(a=n,0===Q&&re(C));else i=n;(s=i!==n?e.substring(s,V):i)!==n&&(i=ae())!==n?(U=t,((o=+(o=s))<1||o>16)&&X("MIDI channel is out of range (1-16)"),t={command:"set_midi_channel",channel:o}):(V=t,t=n)}else V=t,t=n;else V=t,t=n;else V=t,t=n;var o;return t}()),t}if((r=a())!==n&&V===e.length)return r;throw r!==n&&V<e.length&&re({type:"end"}),se(K,J<e.length?e.charAt(J):null,J<e.length?te(J,J+1):te(J,J))}}},function(e,t,r){"use strict";r.r(t);var s=r(0);function n(e,t){let r=0,n=480;t&&t.timebase&&(n=t.timebase);const i=[],a=s.parse(e+";");let o=0;for(let e=0;e<a.length;e++)if(i.push(f(a[e])),o++,o>15)throw new Error("Exceeded maximum MIDI channel (16)");const c=a.length>1?1:0;let h=[77,84,104,100];function l(e){h.push(e>>8&255,255&e)}function u(e){h.push(e>>24&255,e>>16&255,e>>8&255,255&e)}u(6),l(c),l(a.length),l(n);for(let e=0;e<a.length;e++)h.push(77,84,114,107),u(i[e].length),h=h.concat(i[e]);return t&&(t.startTick=r),new Uint8Array(h);function f(e){let t=[],s=n,i=0,a=0;let c=4,h=100,l=6,u=0,f=0;function d(...e){t=t.concat(e)}function p(e){throw new Error(`${e}`)}function g(e,t){let r=s;e&&(r=4*n/e);let i=r;for(let e=0;e<t;e++)i/=2,r+=i;return r}function m(e){(e<0||e>268435455)&&p("illegal length");const t=[];do{t.push(127&e),e>>>=7}while(e>0);for(;t.length>0;){let e=t.pop();t.length>0&&(e|=128),d(e)}}for(;f<e.length;){const t=e[f];switch(t.command){case"note":{let r=12*(c+1)+[9,11,0,2,4,5,7]["abcdefg".indexOf(t.tone)]+u;for(let e=0;e<t.accidentals.length;e++)"+"===t.accidentals[e]&&r++,"-"===t.accidentals[e]&&r--;(r<0||r>127)&&p("illegal note number (0-127)");let s=g(t.length,t.dots.length);for(;e[f+1]&&"tie"===e[f+1].command;)f++,s+=g(e[f].length,e[f].dots.length);const n=Math.round(s*l/8);m(a),d(144|o,r,h),m(n),d(128|o,r,0),a=s-n,i+=s;break}case"rest":let n=g(t.length,t.dots.length);a+=n,i+=n;break;case"octave":c=t.number;break;case"octave_up":c++;break;case"octave_down":c--;break;case"note_length":s=g(t.length,t.dots.length);break;case"gate_time":l=t.quantity;break;case"velocity":h=t.value;break;case"volume":m(a),d(176|o,7,t.value);break;case"pan":m(a),d(176|o,10,t.value+64);break;case"expression":m(a),d(176|o,11,t.value);break;case"control_change":m(a),d(176|o,t.number,t.value);break;case"program_change":m(a),d(192|o,t.number);break;case"channel_aftertouch":m(a),d(208|o,t.value);break;case"tempo":{const e=6e7/t.value;(e<1||e>16777215)&&p("illegal tempo"),m(a),d(255,81,3,e>>16&255,e>>8&255,255&e);break}case"start_point":r=i;break;case"key_shift":u=t.value;break;case"set_midi_channel":o=t.channel-1}(c<-1||c>10)&&p("octave is out of range"),f++}return t}}class i{static clear(){"undefined"!=typeof document&&(document.getElementById("debug").innerHTML="")}static log(e){if("undefined"==typeof document)return;const t=document.getElementById("debug");if(t){const r=document.createElement("div"),s=document.createTextNode(e);for(r.appendChild(s),t.appendChild(r);t.scrollHeight>t.clientHeight;)t.removeChild(t.firstChild)}}}class a{constructor(e,t=1024){this.synthesizer=e,this.bufferSize=t;try{this.context=window.AudioContext?new AudioContext:new webkitAudioContext}catch(e){return void i.log("error: This browser does not support Web Audio API.")}this.bufferL=new Float32Array(this.bufferSize),this.bufferR=new Float32Array(this.bufferSize),this.scriptProcessor=this.context.createScriptProcessor(this.bufferSize,0,2),this.scriptProcessor.onaudioprocess=e=>this.process(e),this.scriptProcessor.connect(this.context.destination),window.savedReference=this.scriptProcessor,i.log("  Sampling rate : "+this.context.sampleRate+" Hz"),i.log("  Buffer size   : "+this.scriptProcessor.bufferSize+" samples")}process(e){const t=e.outputBuffer.getChannelData(0),r=e.outputBuffer.getChannelData(1);this.synthesizer.render(this.bufferL,this.bufferR,this.context.sampleRate);for(let e=0;e<this.bufferSize;e++)t[e]=this.bufferL[e],r[e]=this.bufferR[e]}}class o{static random(e,t){return e+Math.random()*(t-e)}static clamp(e,t,r){if(t>r){var s=t;t=r,r=s}return e<t?t:e>r?r:e}static linearMap(e,t,r,s,n){return s+(e-t)*(n-s)/(r-t)}static clampedLinearMap(e,t,r,s,n){return this.clamp(this.linearMap(e,t,r,s,n),s,n)}static ease(e,t,r,s){return e+(t-e)*(1-Math.exp(-r*s))}static radian(e){return.0174532925199433*e}static degree(e){return 57.29577951308232*e}static wrap(e,t,r){const s=(e-t)%(r-t);return s>=0?s+t:s+r}}class c{getSample(e){return e%1<.5?1:-1}}class h{getSample(e){const t=e%1;return t<.25?o.linearMap(t,0,.25,0,1):t<.75?o.linearMap(t,.25,.75,1,-1):o.linearMap(t,.75,1,-1,0)}}class l{constructor(e){this.synthesizer=e,this.state=0}play(e,t){this.state=3,this.note=e,this.frequency=440*Math.pow(2,(e-69)/12),this.volume=t/127,this.phase=0,this.oscillator=new c,this.vibratoOscillator=new h,this.vibratoPhase=0,this.vibratoFrequency=8,this.vibratoAmplitude=.5,this.oversampling=4}stop(){this.state=4}render(e,t,r){if(0!==this.state)for(let s=0;s<t;s++){const t=this.synthesizer.modulationWheel*this.vibratoAmplitude,n=r/this.vibratoFrequency;this.vibratoPhase+=1/n;const i=this.vibratoOscillator.getSample(this.vibratoPhase)*t,a=r/this.note2frequency(this.note+this.synthesizer.pitchBend+i);let o=0;for(let e=0;e<this.oversampling;e++)o+=this.oscillator.getSample(this.phase),this.phase+=1/a/this.oversampling;if(e[s]+=o/this.oversampling*this.volume*.1,4===this.state?this.volume-=.005:this.volume*=.99999,this.volume<0)return void(this.state=0)}}isPlaying(){return 0!==this.state}note2frequency(e){return 440*Math.pow(2,(e-69)/12)}}class u{reset(){this.voices=[];for(let e=0;e<32;e++)this.voices[e]=new l(this);this.keyState=[],this.volume=100,this.pan=64,this.expression=127,this.damperPedal=!1,this.pitchBend=0,this.modulationWheel=0,this.channelBuffer=new Float32Array(4096)}noteOn(e,t){this.keyState[e]=!0;for(let t=0;t<32;t++)this.voices[t].isPlaying()&&this.voices[t].note===e&&this.voices[t].stop();for(let r=0;r<32;r++)if(!this.voices[r].isPlaying()){this.voices[r].play(e,t);break}}noteOff(e,t){if(this.keyState[e]=!1,!this.damperPedal)for(let t=0;t<32;t++)this.voices[t].isPlaying()&&this.voices[t].note===e&&this.voices[t].stop()}allNotesOff(){for(let e=0;e<32;e++)this.voices[e].isPlaying()&&this.voices[e].stop()}damperPedalOn(){this.damperPedal=!0}damperPedalOff(){this.damperPedal=!1;for(let e=0;e<32;e++)!1===this.keyState[this.voices[e].note]&&this.voices[e].stop()}programChange(e){}setPitchBend(e){this.pitchBend=2*e/8192}setModulationWheel(e){this.modulationWheel=e/127}setVolume(e){this.volume=e}setPan(e){this.pan=e}setExpression(e){this.expression=e}render(e,t,r){for(let t=0;t<e.length;t++)this.channelBuffer[t]=0;for(let t=0;t<32;t++)this.voices[t].render(this.channelBuffer,e.length,r);const s=this.volume/127*(this.expression/127),n=s*o.clampedLinearMap(this.pan,64,127,1,0),i=s*o.clampedLinearMap(this.pan,0,64,0,1);for(let r=0;r<e.length;r++)e[r]+=this.channelBuffer[r]*n,t[r]+=this.channelBuffer[r]*i}}class f{constructor(e){this.options=e,this.channels=[];for(let e=0;e<16;e++)this.channels[e]=new u;this.reset(),this.audioManager=null,class{static isiOS(){return this.isiPhone()||this.isiPad()}static isiPhone(){return"undefined"!=typeof document&&window.navigator.userAgent.indexOf("iPhone")>=0}static isiPad(){return"undefined"!=typeof document&&window.navigator.userAgent.indexOf("iPad")>=0}}.isiOS()||this.createAudioManager()}createAudioManager(){this.audioManager||(i.log("Initializing Web Audio"),this.audioManager=new a(this))}reset(){i.log("Initializing Synthesizer");for(let e=0;e<16;e++)this.channels[e].reset()}render(e,t,r){for(let r=0;r<e.length;r++)e[r]=0,t[r]=0;for(let s=0;s<16;s++)this.channels[s].render(e,t,r)}processMIDIMessage(e){if(!e)return;this.createAudioManager();const t=e[0];if(!t)return;const r=t>>4,s=15&t,n=s+1;if(9===r){const t=e[1],r=e[2];this.log(`Ch. ${n} Note On  note: ${t} velocity: ${r}`),this.channels[s].noteOn(t,r)}if(8===r){const t=e[1],r=e[2];this.log(`Ch. ${n} Note Off note: ${t} velocity: ${r}`),this.channels[s].noteOff(t,r)}if(12===r){const t=e[1];this.log(`Ch. ${n} Program Change: ${t}`),this.channels[s].programChange(t)}if(14===r){const t=e[1],r=(e[2]<<7|t)-8192;this.log(`Ch. ${n} Pitch bend: ${r}`),this.channels[s].setPitchBend(r)}if(11===r){const t=e[1],r=e[2];1===t&&(this.log(`Ch. ${n} Modulation Wheel: ${r}`),this.channels[s].setModulationWheel(r)),7===t&&(this.log(`Ch. ${n} Channel Volume: ${r}`),this.channels[s].setVolume(r)),10===t&&(this.log(`Ch. ${n} Pan: ${r}`),this.channels[s].setPan(r)),11===t&&(this.log(`Ch. ${n} Expression Controller: ${r}`),this.channels[s].setExpression(r)),64===t&&(r>=64?(this.log(`Ch. ${n} Damper Pedal On`),this.channels[s].damperPedalOn()):(this.log(`Ch. ${n} Damper Pedal Off`),this.channels[s].damperPedalOff())),123===t&&0===r&&(this.log(`Ch. ${n} All Notes Off`),this.channels[s].allNotesOff())}}log(e){this.options&&this.options.verbose&&i.log(e)}}class d{constructor(e,t,r){this.player=e,this.pos=t,this.endPos=t+r,this.finished=!1,this.nextEventTick=this.readDeltaTick()}update(e,t){if(!this.finished)for(;this.nextEventTick<e;){const e=this.readByte(),r=e>>4;if(255===e){const e=this.readByte(),t=this.readByte();if(81===e){if(3===t){const e=this.readByte()<<16|this.readByte()<<8|this.readByte();this.player.quarterTime=e/1e3}}else this.pos+=t}if(240===e){const t=[e];for(;;){if(this.pos>=this.endPos)throw Error("illegal system exlusive message");const e=this.readByte();if(247===e)break;t.push(e)}this.player.synthesizer.processMIDIMessage(t)}switch(241!==e&&242!==e&&243!==e||this.readByte(),r){case 8:case 9:case 10:case 11:case 14:{const s=this.readByte(),n=this.readByte();(!t||8!==r&&9!==r)&&this.player.synthesizer.processMIDIMessage([e,s,n]);break}case 12:case 13:{const t=this.readByte();this.player.synthesizer.processMIDIMessage([e,t]);break}}if(this.pos>=this.endPos){this.finished=!0;break}this.nextEventTick+=this.readDeltaTick()}}readByte(){return this.player.smf[this.pos++]}readDeltaTick(){let e,t=0;do{e=this.readByte(),t<<=7,t|=127&e}while(128&e);if(t>268435455)throw new Error("illegal delta tick");return t}}class p{constructor(e){this.synthesizer=e}play(e,t=0){this.smf=e,this.startTick=t,this.quarterTime=500;let r=8;function s(){return e[r++]<<8|e[r++]}const n=s();this.trackNumber=s(),this.timebase=s();const i=[77,84,104,100,0,0,0,6];for(let e=0;e<i.length;e++)if(this.smf[e]!==i[e])throw new Error("not a standard MIDI file");if(0!==n&&1!==n)throw new Error("wrong SMF format");if(0===n&&1!==this.trackNumber)throw new Error("illegal track number");this.tracks=[];for(let t=0;t<this.trackNumber;t++){r+=4;const t=e[r++]<<24|e[r++]<<16|e[r++]<<8|e[r++];this.tracks.push(new d(this,r,t)),r+=t}this.prevTime=Date.now(),this.currentTick=0,this.intervalId||(this.intervalId=setInterval(()=>this.onInterval(),.01))}stop(){clearInterval(this.intervalId),this.intervalId=null}onInterval(){const e=Date.now(),t=e-this.prevTime;this.prevTime=e;const r=this.quarterTime/this.timebase;let s=!1;this.currentTick<this.startTick?(this.currentTick=this.startTick,s=!0):this.currentTick+=t/r;for(let e=0;e<this.tracks.length;e++)this.tracks[e].update(this.currentTick,s);let n=0;for(let e=0;e<this.tracks.length;e++)!1===this.tracks[e].finished&&n++;0===n&&this.stop()}}r.d(t,"mml2smf",(function(){return n})),r.d(t,"Synthesizer",(function(){return f})),r.d(t,"SMFPlayer",(function(){return p}))}])}));