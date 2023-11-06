using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface
import HiGHS


# For bug hunting:
seed = rand(UInt64)
seed = 0xd038b7eafdcb28c1
@show seed
#seed = 0xeadb922ca734998b  
Random.seed!(seed)

# TROUBLESOME SEED seed = 0x8750860d6fd5025f -> NEEDS TO BE CHECK AGAIN!

n = 20
#const ri = rand(n)
#const ai = rand(n)
#const Ωi = rand(Float64)
#const bi = sum(ai)
#Ai = randn(n, n)
#Ai = Ai' * Ai
#const Mi = (Ai + Ai') / 2
#@assert isposdef(Mi)

#@show ri
#@show ai
#@show Ωi 
#@show bi
#@show Ai
#@show Mi

const ri = [0.045054599902509596, 0.15640436665619184, 0.2735669741522424, 0.2784384724840858, 0.7273512458022291, 0.7259501747199306, 0.22719857001620958, 0.7488803015182965, 0.7551865762729075, 0.8729424803227761, 0.1031509613456203, 0.5883209711669155, 0.5222239085099922, 0.45356565475023625, 0.12071201005492083, 0.2826881085023156, 0.26781345864628303, 0.6719733143780509, 0.4972650694624705, 0.34719664179834164]

const ai = [0.9180595602471104, 0.6223508120192558, 0.7826711851406587, 0.8877891199455334, 0.31014726178572927, 0.2399447346410375, 0.7568595790126875, 0.017927403665631836, 0.05683421070174455, 0.15282271111057755, 0.125154806480549, 0.06918870150741196, 0.3219811750202961, 0.949408167695982, 0.3547498287597832, 0.02808017754876091, 0.930299590021576, 0.0009707781039260954, 0.6897214878434292, 0.7538246122510148]

const Ωi = 0.7014455397435948
#bi = 8.968785903502694
const bi = sum(ai)

Ai = [21.067822898841943 -0.2456591665304344 0.41724237018201815 0.2239312963112271 -0.7373880046845093 -3.267709286092762 1.2558960614915409 -6.537124595164123 2.560502965591387 0.812267330132679 -9.606392059140921 -4.391021450539187 1.0922662708120585 0.24790820343248382 -1.8201734143546722 -1.8415302653951926 4.567113068825415 5.645481991144209 -2.3873485872566453 1.8015299584725635; -0.2456591665304344 20.007108977419858 -2.5536460373492655 5.386746979815513 -0.45695463200069586 5.1077700321396735 -2.3519215972671277 5.251859978684248 1.4839692307383068 -0.33174216510599464 0.11908657953473512 0.6716004021210218 0.9704233608273368 -6.9154323685373384 -4.833342524281925 -0.2642673928906647 5.629609027127837 6.297468737216921 4.120679687210085 12.27738844183925; 0.41724237018201815 -2.5536460373492655 26.251375897756276 4.732490313443435 -0.03188432344993197 3.9330809903829964 3.4379369530243302 1.9872826405433353 -9.14693567668028 1.3777248881934414 4.1375305929784805 2.620712773292841 -2.7469464954968563 -5.231991290099736 6.353298857440429 -0.955953934131732 8.08225555417372 -5.918405393623921 -6.096759128605076 -1.9361648561616194; 0.2239312963112271 5.386746979815513 4.732490313443435 25.590782695097285 -0.31324620373276135 9.244426427136773 8.255280707380486 -1.1454702860098893 3.9450651929238583 -0.9807217330998804 10.445630671621357 -5.416861940264175 1.0850574513395985 6.405258216592411 0.7226836884734482 -4.8838038803785455 -4.611551533620946 3.873245876877796 -6.980725332973074 10.201493534122653; -0.7373880046845093 -0.45695463200069586 -0.03188432344993197 -0.31324620373276135 27.9299036429962 8.68867111135701 -12.668975887413147 2.5211329932122455 2.7623200478482346 1.3480142672201076 -2.7003130970822617 10.26154233613517 4.71252610285813 -4.165112055049683 2.6690353726933287 2.5852742817386596 -0.06437943294211701 -0.6224829336869455 5.829240592848388 -1.9283235560642311; -3.267709286092762 5.1077700321396735 3.9330809903829964 9.244426427136773 8.68867111135701 16.539761343969953 -5.272052298662209 2.7061353420778134 8.821710432108043 -3.429483606674073 1.7535895899297458 4.044916031316452 3.4114820972570588 -5.40445109873675 5.097142722451032 -1.2374133950901256 1.944760684957939 2.850488218102691 1.6720203884477016 -0.4401050512934188; 1.2558960614915409 -2.3519215972671277 3.4379369530243302 8.255280707380486 -12.668975887413147 -5.272052298662209 32.05886607071271 -3.9153443574171445 0.597311808702748 -7.159939393209872 -0.12314777431526742 -13.67686715031358 -8.926154333134136 2.663285783794903 -6.19907990632074 -6.962150847403804 3.7699108227938423 -7.7567604353040815 -1.2567381540418687 8.84771792529003; -6.537124595164123 5.251859978684248 1.9872826405433353 -1.1454702860098893 2.5211329932122455 2.7061353420778134 -3.9153443574171445 18.39876360622575 3.665158869432717 1.9908664493145114 1.0029510365299954 4.832189958258888 -2.3288329628958997 0.1276673220863505 4.377010735421336 7.763425880289378 4.580656874218642 -0.591498789421499 8.102008910356536 4.534697004728115; 2.560502965591387 1.4839692307383068 -9.14693567668028 3.9450651929238583 2.7623200478482346 8.821710432108043 0.597311808702748 3.665158869432717 25.540378585492927 -7.64389396065181 -6.468799238951818 -3.04588024923645 -6.881731562872513 0.20525279177233202 -2.453794913853438 5.142171006779962 0.10012025630256935 3.272979478984822 7.525702636596447 0.9292970275171553; 0.812267330132679 -0.33174216510599464 1.3777248881934414 -0.9807217330998804 1.3480142672201076 -3.429483606674073 -7.159939393209872 1.9908664493145114 -7.64389396065181 17.770119811755308 3.2276326021259343 -0.45021511936117437 5.256501578652799 -2.483677462289625 -5.277695539374783 -10.030697060234619 1.5729721060304087 1.5609060749512498 -7.883238394580825 2.5061054391405833; -9.606392059140921 0.11908657953473512 4.1375305929784805 10.445630671621357 -2.7003130970822617 1.7535895899297458 -0.12314777431526742 1.0029510365299954 -6.468799238951818 3.2276326021259343 23.311829677236993 -3.588829979556081 0.6635847539879597 6.229360364819488 -0.8035993980326457 3.1191879904878586 -10.345306129667012 2.901762241411971 -10.83293173145159 7.625555131416707; -4.391021450539187 0.6716004021210218 2.620712773292841 -5.416861940264175 10.26154233613517 4.044916031316452 -13.67686715031358 4.832189958258888 -3.04588024923645 -0.45021511936117437 -3.588829979556081 19.82169467180281 4.073442728153711 -3.376027047776719 5.27635312754488 2.2021586127011497 4.602366235060449 -3.14763607501543 7.6731283676994115 -9.003655004313524; 1.0922662708120585 0.9704233608273368 -2.7469464954968563 1.0850574513395985 4.71252610285813 3.4114820972570588 -8.926154333134136 -2.3288329628958997 -6.881731562872513 5.256501578652799 0.6635847539879597 4.073442728153711 16.89214082514548 2.903567286178969 5.0849418958348735 0.515061950019791 -1.2044452507741277 1.3264038737633472 -6.113660031427819 -9.286940993467605; 0.24790820343248382 -6.9154323685373384 -5.231991290099736 6.405258216592411 -4.165112055049683 -5.40445109873675 2.663285783794903 0.1276673220863505 0.20525279177233202 -2.483677462289625 6.229360364819488 -3.376027047776719 2.903567286178969 35.61584093296939 -1.9494384494934036 13.921083869621652 -10.247364774410546 -7.406062816640134 -0.10711300217400721 -9.727649222050019; -1.8201734143546722 -4.833342524281925 6.353298857440429 0.7226836884734482 2.6690353726933287 5.097142722451032 -6.19907990632074 4.377010735421336 -2.453794913853438 -5.277695539374783 -0.8035993980326457 5.27635312754488 5.0849418958348735 -1.9494384494934036 28.85361889354387 0.7281434872766035 -9.486215961970576 0.14167249041049612 -7.047692104013578 -5.583710300280806; -1.8415302653951926 -0.2642673928906647 -0.955953934131732 -4.8838038803785455 2.5852742817386596 -1.2374133950901256 -6.962150847403804 7.763425880289378 5.142171006779962 -10.030697060234619 3.1191879904878586 2.2021586127011497 0.515061950019791 13.921083869621652 0.7281434872766035 36.15931261936261 -1.8337579067748282 -0.21715857519199916 7.095866868366173 -11.356399337040415; 4.567113068825415 5.629609027127837 8.08225555417372 -4.611551533620946 -0.06437943294211701 1.944760684957939 3.7699108227938423 4.580656874218642 0.10012025630256935 1.5729721060304087 -10.345306129667012 4.602366235060449 -1.2044452507741277 -10.247364774410546 -9.486215961970576 -1.8337579067748282 23.352985852121297 -4.0851885676116995 9.077805998772966 3.099810970073503; 5.645481991144209 6.297468737216921 -5.918405393623921 3.873245876877796 -0.6224829336869455 2.850488218102691 -7.7567604353040815 -0.591498789421499 3.272979478984822 1.5609060749512498 2.901762241411971 -3.14763607501543 1.3264038737633472 -7.406062816640134 0.14167249041049612 -0.21715857519199916 -4.0851885676116995 22.173849558437485 1.6985945415698833 5.991922716154; -2.3873485872566453 4.120679687210085 -6.096759128605076 -6.980725332973074 5.829240592848388 1.6720203884477016 -1.2567381540418687 8.102008910356536 7.525702636596447 -7.883238394580825 -10.83293173145159 7.6731283676994115 -6.113660031427819 -0.10711300217400721 -7.047692104013578 7.095866868366173 9.077805998772966 1.6985945415698833 25.63410001961338 -2.1631643217274314; 1.8015299584725635 12.27738844183925 -1.9361648561616194 10.201493534122653 -1.9283235560642311 -0.4401050512934188 8.84771792529003 4.534697004728115 0.9292970275171553 2.5061054391405833 7.625555131416707 -9.003655004313524 -9.286940993467605 -9.727649222050019 -5.583710300280806 -11.356399337040415 3.099810970073503 5.991922716154 -2.1631643217274314 35.87523667318088]
#Mi = [21.067822898841943 -0.2456591665304344 0.41724237018201815 0.2239312963112271 -0.7373880046845093 -3.267709286092762 1.2558960614915409 -6.537124595164123 2.560502965591387 0.812267330132679 -9.606392059140921 -4.391021450539187 1.0922662708120585 0.24790820343248382 -1.8201734143546722 -1.8415302653951926 4.567113068825415 5.645481991144209 -2.3873485872566453 1.8015299584725635; -0.2456591665304344 20.007108977419858 -2.5536460373492655 5.386746979815513 -0.45695463200069586 5.1077700321396735 -2.3519215972671277 5.251859978684248 1.4839692307383068 -0.33174216510599464 0.11908657953473512 0.6716004021210218 0.9704233608273368 -6.9154323685373384 -4.833342524281925 -0.2642673928906647 5.629609027127837 6.297468737216921 4.120679687210085 12.27738844183925; 0.41724237018201815 -2.5536460373492655 26.251375897756276 4.732490313443435 -0.03188432344993197 3.9330809903829964 3.4379369530243302 1.9872826405433353 -9.14693567668028 1.3777248881934414 4.1375305929784805 2.620712773292841 -2.7469464954968563 -5.231991290099736 6.353298857440429 -0.955953934131732 8.08225555417372 -5.918405393623921 -6.096759128605076 -1.9361648561616194; 0.2239312963112271 5.386746979815513 4.732490313443435 25.590782695097285 -0.31324620373276135 9.244426427136773 8.255280707380486 -1.1454702860098893 3.9450651929238583 -0.9807217330998804 10.445630671621357 -5.416861940264175 1.0850574513395985 6.405258216592411 0.7226836884734482 -4.8838038803785455 -4.611551533620946 3.873245876877796 -6.980725332973074 10.201493534122653; -0.7373880046845093 -0.45695463200069586 -0.03188432344993197 -0.31324620373276135 27.9299036429962 8.68867111135701 -12.668975887413147 2.5211329932122455 2.7623200478482346 1.3480142672201076 -2.7003130970822617 10.26154233613517 4.71252610285813 -4.165112055049683 2.6690353726933287 2.5852742817386596 -0.06437943294211701 -0.6224829336869455 5.829240592848388 -1.9283235560642311; -3.267709286092762 5.1077700321396735 3.9330809903829964 9.244426427136773 8.68867111135701 16.539761343969953 -5.272052298662209 2.7061353420778134 8.821710432108043 -3.429483606674073 1.7535895899297458 4.044916031316452 3.4114820972570588 -5.40445109873675 5.097142722451032 -1.2374133950901256 1.944760684957939 2.850488218102691 1.6720203884477016 -0.4401050512934188; 1.2558960614915409 -2.3519215972671277 3.4379369530243302 8.255280707380486 -12.668975887413147 -5.272052298662209 32.05886607071271 -3.9153443574171445 0.597311808702748 -7.159939393209872 -0.12314777431526742 -13.67686715031358 -8.926154333134136 2.663285783794903 -6.19907990632074 -6.962150847403804 3.7699108227938423 -7.7567604353040815 -1.2567381540418687 8.84771792529003; -6.537124595164123 5.251859978684248 1.9872826405433353 -1.1454702860098893 2.5211329932122455 2.7061353420778134 -3.9153443574171445 18.39876360622575 3.665158869432717 1.9908664493145114 1.0029510365299954 4.832189958258888 -2.3288329628958997 0.1276673220863505 4.377010735421336 7.763425880289378 4.580656874218642 -0.591498789421499 8.102008910356536 4.534697004728115; 2.560502965591387 1.4839692307383068 -9.14693567668028 3.9450651929238583 2.7623200478482346 8.821710432108043 0.597311808702748 3.665158869432717 25.540378585492927 -7.64389396065181 -6.468799238951818 -3.04588024923645 -6.881731562872513 0.20525279177233202 -2.453794913853438 5.142171006779962 0.10012025630256935 3.272979478984822 7.525702636596447 0.9292970275171553; 0.812267330132679 -0.33174216510599464 1.3777248881934414 -0.9807217330998804 1.3480142672201076 -3.429483606674073 -7.159939393209872 1.9908664493145114 -7.64389396065181 17.770119811755308 3.2276326021259343 -0.45021511936117437 5.256501578652799 -2.483677462289625 -5.277695539374783 -10.030697060234619 1.5729721060304087 1.5609060749512498 -7.883238394580825 2.5061054391405833; -9.606392059140921 0.11908657953473512 4.1375305929784805 10.445630671621357 -2.7003130970822617 1.7535895899297458 -0.12314777431526742 1.0029510365299954 -6.468799238951818 3.2276326021259343 23.311829677236993 -3.588829979556081 0.6635847539879597 6.229360364819488 -0.8035993980326457 3.1191879904878586 -10.345306129667012 2.901762241411971 -10.83293173145159 7.625555131416707; -4.391021450539187 0.6716004021210218 2.620712773292841 -5.416861940264175 10.26154233613517 4.044916031316452 -13.67686715031358 4.832189958258888 -3.04588024923645 -0.45021511936117437 -3.588829979556081 19.82169467180281 4.073442728153711 -3.376027047776719 5.27635312754488 2.2021586127011497 4.602366235060449 -3.14763607501543 7.6731283676994115 -9.003655004313524; 1.0922662708120585 0.9704233608273368 -2.7469464954968563 1.0850574513395985 4.71252610285813 3.4114820972570588 -8.926154333134136 -2.3288329628958997 -6.881731562872513 5.256501578652799 0.6635847539879597 4.073442728153711 16.89214082514548 2.903567286178969 5.0849418958348735 0.515061950019791 -1.2044452507741277 1.3264038737633472 -6.113660031427819 -9.286940993467605; 0.24790820343248382 -6.9154323685373384 -5.231991290099736 6.405258216592411 -4.165112055049683 -5.40445109873675 2.663285783794903 0.1276673220863505 0.20525279177233202 -2.483677462289625 6.229360364819488 -3.376027047776719 2.903567286178969 35.61584093296939 -1.9494384494934036 13.921083869621652 -10.247364774410546 -7.406062816640134 -0.10711300217400721 -9.727649222050019; -1.8201734143546722 -4.833342524281925 6.353298857440429 0.7226836884734482 2.6690353726933287 5.097142722451032 -6.19907990632074 4.377010735421336 -2.453794913853438 -5.277695539374783 -0.8035993980326457 5.27635312754488 5.0849418958348735 -1.9494384494934036 28.85361889354387 0.7281434872766035 -9.486215961970576 0.14167249041049612 -7.047692104013578 -5.583710300280806; -1.8415302653951926 -0.2642673928906647 -0.955953934131732 -4.8838038803785455 2.5852742817386596 -1.2374133950901256 -6.962150847403804 7.763425880289378 5.142171006779962 -10.030697060234619 3.1191879904878586 2.2021586127011497 0.515061950019791 13.921083869621652 0.7281434872766035 36.15931261936261 -1.8337579067748282 -0.21715857519199916 7.095866868366173 -11.356399337040415; 4.567113068825415 5.629609027127837 8.08225555417372 -4.611551533620946 -0.06437943294211701 1.944760684957939 3.7699108227938423 4.580656874218642 0.10012025630256935 1.5729721060304087 -10.345306129667012 4.602366235060449 -1.2044452507741277 -10.247364774410546 -9.486215961970576 -1.8337579067748282 23.352985852121297 -4.0851885676116995 9.077805998772966 3.099810970073503; 5.645481991144209 6.297468737216921 -5.918405393623921 3.873245876877796 -0.6224829336869455 2.850488218102691 -7.7567604353040815 -0.591498789421499 3.272979478984822 1.5609060749512498 2.901762241411971 -3.14763607501543 1.3264038737633472 -7.406062816640134 0.14167249041049612 -0.21715857519199916 -4.0851885676116995 22.173849558437485 1.6985945415698833 5.991922716154; -2.3873485872566453 4.120679687210085 -6.096759128605076 -6.980725332973074 5.829240592848388 1.6720203884477016 -1.2567381540418687 8.102008910356536 7.525702636596447 -7.883238394580825 -10.83293173145159 7.6731283676994115 -6.113660031427819 -0.10711300217400721 -7.047692104013578 7.095866868366173 9.077805998772966 1.6985945415698833 25.63410001961338 -2.1631643217274314; 1.8015299584725635 12.27738844183925 -1.9361648561616194 10.201493534122653 -1.9283235560642311 -0.4401050512934188 8.84771792529003 4.534697004728115 0.9292970275171553 2.5061054391405833 7.625555131416707 -9.003655004313524 -9.286940993467605 -9.727649222050019 -5.583710300280806 -11.356399337040415 3.099810970073503 5.991922716154 -2.1631643217274314 35.87523667318088]

const Mi = (Ai + Ai') / 2
@assert isposdef(Mi)

function prepare_portfolio_lmo()
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    I = collect(1:n)
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        if i in I
            MOI.add_constraint(o, x[i], MOI.Integer())
        end
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai, x), 0.0),
        MOI.LessThan(bi),
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.GreaterThan(1.0),
    )
    lmo = FrankWolfe.MathOptLMO(o)
    return lmo
end

function f(x)
    return 1 / 2 * Ωi * dot(x, Mi, x) - dot(ri, x)
end
function grad!(storage, x)
    mul!(storage, Mi, x, Ωi, 0)
    storage .-= ri
    return storage
end

@testset "Portfolio strong branching" begin
    lmo = prepare_portfolio_lmo()
    x, _, result_baseline = Boscia.solve(f, grad!, lmo, verbose=true)
    @test dot(ai, x) <= bi + 1e-6
    @test f(x) <= f(result_baseline[:raw_solution]) + 1e-6

    blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
    branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
    MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), false)

    lmo = prepare_portfolio_lmo()
    x, _, result_strong_branching =
        Boscia.solve(f, grad!, lmo, verbose=true, branching_strategy=branching_strategy)

    @test dot(ai, x) <= bi + 1e-6
    @test f(x) <= f(result_baseline[:raw_solution]) + 1e-6
end

#plot(result_baseline[:list_time],result_baseline[:list_ub], label="BL"); plot!(result_baseline[:list_time],result_baseline[:list_lb], label="BL")
#plot!(result_strong_branching[:list_time], result_strong_branching[:list_ub], label="SB"); plot!(result_strong_branching[:list_time], result_strong_branching[:list_lb], label="SB")

#plot(result_baseline[:list_ub], label="BL")
#plot!(result_baseline[:list_lb], label="BL")
#plot!(result_strong_branching[:list_ub], label="SB")
#plot!(result_strong_branching[:list_lb], label="SB")
