import numpy as np
import casadi as cs
import time
import matplotlib.pyplot as plt

import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parent/Path("Python"))]


#     ____
#    |    |____
#   |      ____
#    |    | condenser
#     |__|
#     |__|
#     |__|
#     |__| rectifying section
#     |__|
#     |__|
# ____|__|
# ____ __|
# feed|__|
#     |__|
#     |__|
#     |__| stripping section
#     |__|
#     |__|
#    |    |____
#   |      ____
#    |____| reboiler


#Molar Methanol concentrations + molar tray holdups
nx = 42 + 40  # 42 (x_Me) + 40 (n_holdup)

#Liquid molar fluxes, vapor molar fluxes, temperatures
nz = 40 + 40 + 42 # 40 (L) + 40 (V) + 42 (T)

#Volumetric flow reflux Q and boiler heat input Q
nu = 2

nparam = 17

model_params = {
    'nvolK': 1.5458567140000001E-01,    #Tray K reference volume
    'nvolM': 1.7499999999999999E-01,  #Feed tray reference volume
    # 'nvolM': 1.5458567140000001E-01,    #Feed tray reference volume
    
    'alphastrip': 6.1895708603484367E-01, #Tray efficiency, stripping section
    'alpharect': 3.4717208398678062E-01,  #Tray efficiency, rectifying section
    'flowwidth': 1.6593025789999999E-01,
    'Qloss': 5.0695122527590109E-01,     #Boiler heat loss
    'nvolB': 8.5000000000000000E+00,    #Reboiler volume holdup
    'nvolD': 1.7000000000000001E-01,    #Distillate volume holdup
    'Ptop': 9.3885430857029321E+04,     #Condenser pressure
    'DPstrip': 2.5000000000000000E+02,  #Pressure difference, stripping section
    'DPrect': 1.8760537088149468E+02,
    # 'DPrect': 2.5000000000000000E+02,    #Pressure difference, rectifying section
    
    'F_lh': 1.4026000000000000E+01,     #Molar feed flow
    'xFeed': 3.2000000000000001E-01,    #Feed Methanol ratio
    'TfeedC': 7.1054000000000002E+01,   #Feed temperature (Celsius)
    'TcondC': 4.7163089489100003E+01,   #Condenser temperature (Celsius)
    'L_ref': 4.1833910753991770E+00,    #Reference reflux
    'Q_ref': 2.4899344810136301E+00,    #Reference heating
    }

u_init = [4.1833910982822058E+00, 2.4899344742988991E+00]
lb_u = [1.0, 1.0]
ub_u = [7.0, 5.0]


x_init = \
[0.2193611617799063,
 0.3336302862386372,
 0.3731313325062595,
 0.39896472354654333,
 0.41533719381260475,
 0.4254839937228718,
 0.4316837935421362,
 0.43543569751236455,
 0.4376891864721443,
 0.43903262905928286,
 0.43982597315656735,
 0.4402877497904797,
 0.4405500251890295,
 0.4406923891700805,
 0.44076272408112094,
 0.44078980543461005,
 0.44079091412311144,
 0.44077642312834125,
 0.44075255679998443,
 0.4407230491123104,
 0.4406901395817392,
 0.6704192618964515,
 0.7351799737575895,
 0.7897597894363141,
 0.8348172515953903,
 0.8712537707738074,
 0.9002727507876772,
 0.923124645363943,
 0.9409695498079861,
 0.9548173126279774,
 0.9655127114536888,
 0.9737440177301049,
 0.980061860721667,
 0.9849010948567534,
 0.9886019477109929,
 0.9914287934200833,
 0.9935860233184747,
 0.9952310563223864,
 0.9964847878570199,
 0.9974398630174197,
 0.9981671609731486,
 0.9987208401428007] +\
[3.863381195673097,
 3.932226049802884,
 3.977196562639253,
 4.006307033386973,
 4.024602684414341,
 4.0358888958821835,
 4.042769039878679,
 4.046930043347702,
 4.049431464802033,
 4.050926756002938,
 4.051814558339763,
 4.05233648463798,
 4.05263839774603,
 4.052808143763277,
 4.052898549113454,
 4.052941351027017,
 4.052955604932446,
 4.0529527471448805,
 4.052939639227801,
 4.052920397049691,
 3.607116495091858,
 3.7583754503438387,
 3.8917148481441974,
 4.009430069874156,
 4.110221672579829,
 4.1944038520620675,
 4.26332751665606,
 4.3188755452109175,
 4.363094790985764,
 4.397962224784139,
 4.425258001249774,
 4.446512894719387,
 4.463001831479197,
 4.475762615001557,
 4.485626009494682,
 4.49324885518085,
 4.499145695962933,
 4.503716811689627,
 4.507271960563973,
 4.510049896978241]
    

x_init_2 = \
[0.0007898819757909115,
 0.0017496509634407759,
 0.0027992571903949974,
 0.004342784491281894,
 0.0066043323769979885,
 0.00989991934557294,
 0.014664251928439336,
 0.02147319716330936,
 0.031046630794181976,
 0.04420486068740363,
 0.061743680224201444,
 0.0842049254023566,
 0.11157305166307535,
 0.14302048369018389,
 0.17688533766130812,
 0.21098725374620048,
 0.24317258179170506,
 0.2718152985258991,
 0.29605267650664546,
 0.3157330666967309,
 0.3312004445650639,
 0.36942603158584475,
 0.4014426930458525,
 0.44023110326252346,
 0.48551340442161467,
 0.5359464260281759,
 0.5894415078674442,
 0.6435496564220227,
 0.695931817403143,
 0.7447287816381606,
 0.788730693499566,
 0.8273610074528788,
 0.8605507664958192,
 0.8885790906487329,
 0.9119277353618566,
 0.9311695026749097,
 0.9468925225199298,
 0.9596543784913268,
 0.9699580131718311,
 0.978242119084811] +\
[0.9848803986585992,
 0.9901857589205203,
 3.395935412171285,
 3.3972006826383563,
 3.3990251057073984,
 3.401704156003664,
 3.4056205957104444,
 3.411308372117987,
 3.4194899007365165,
 3.431097972171238,
 3.4472520559419637,
 3.469142616076647,
 3.4977761391802002,
 3.5335789657214103,
 3.575970187109377,
 3.623140592012331,
 3.6722689892342473,
 3.7201742704172895,
 3.7640996235338147,
 3.8022492939496937,
 3.8339074053894646,
 3.8592300733568505,
 3.05500364438512,
 3.1063375350667215,
 3.169953778029879,
 3.247573542603287,
 3.3386394427336303,
 3.4410459129012794,
 3.551328848157282,
 3.6651993895946484,
 3.7782403312953434,
 3.8865330311890247,
 3.9870642658361213,
 4.077878740127236,
 4.158025873537421,
 4.227381947288156,
 4.286424128155971,
 4.336010925990513,
 4.377199026157455,
 4.411106873488765,
 4.43882307552088,
 4.46135173334851]


lb_x = [-0.01]*42 + [-0.1]*40
ub_x = [1.01]*42 + [10.0]*40


z_init = \
[8.765107914363698,
 8.787106331643232,
 8.789307470367007,
 8.790195454444534,
 8.790123341660648,
 8.789402066178145,
 8.788264121625536,
 8.78686553826272,
 8.785305923281816,
 8.78364723679401,
 8.78192747156961,
 8.780169731778734,
 8.778387997933846,
 8.776590703329116,
 8.774782924103734,
 8.772967710297705,
 8.771146891237429,
 8.769321561547551,
 8.767492373953488,
 8.765659715501714,
 2.782546940341337,
 2.822411112579974,
 2.8351257821612172,
 2.8455862495713884,
 2.8539999172723634,
 2.8606290594307993,
 2.865765380122027,
 2.8696861889639877,
 2.872635275890039,
 2.8748174364382795,
 2.8763998227654772,
 2.877516257684113,
 2.8782724559458406,
 2.878751135583851,
 2.8790165741126224,
 2.8791184656798956,
 2.8790950843473126,
 2.878975824680423,
 2.8787832131565576,
 2.8785344845386325,
 3.7489688386445703,
 3.751169977185859,
 3.752057961126931,
 3.751985848226562,
 3.75126457263124,
 3.75012662796529,
 3.7487280444903774,
 3.747168429400522,
 3.745509742806507,
 3.7437899774766037,
 3.7420322375793442,
 3.740250503627812,
 3.7384532089192324,
 3.7366454295969547,
 3.734830215704193,
 3.733009396568163,
 3.731184066812245,
 3.7293548791598456,
 3.727522220655656,
 3.529587943700007,
 3.569452115807212,
 3.5821667852381145,
 3.592627252480061,
 3.601040920000433,
 3.6076700619776987,
 3.6128063825021988,
 3.616727191204299,
 3.619676278021998,
 3.621858438489323,
 3.623440824754021,
 3.6245572596202216,
 3.62531345783572,
 3.6257921374340887,
 3.6260575759313443,
 3.626159467475165,
 3.626136086125412,
 3.6260168264456856,
 3.6258242149122157,
 3.6255754862872984,
 3.527859634499679,
 87.07518960318062,
 82.54885781313709,
 81.08511671111634,
 80.14069865864266,
 79.53221933296352,
 79.13575198023646,
 78.87029469160841,
 78.68480316978406,
 78.5477571655613,
 78.43991684323566,
 78.34961751239067,
 78.26981551522998,
 78.19626785856451,
 78.12642237442402,
 78.05874528785895,
 77.99231531756456,
 77.92657921669026,
 77.86120474914591,
 77.79599234426986,
 77.73082204166866,
 77.66562164201588,
 71.09496160841573,
 69.44880511620633,
 68.12226139454849,
 67.06079912576944,
 66.21779530802625,
 65.55002743604167,
 65.02043275022177,
 64.59858140061951,
 64.26012546726517,
 63.98589335264476,
 63.760944540013256,
 63.57371542885716,
 63.41529778917354,
 63.27885129423076,
 63.15913560929375,
 63.05214283589952,
 62.954811408060124,
 62.86480476464194,
 62.78034086892876,
 62.700061296306565,
 62.62293092969283
 ]

z_init_2 = \
[8.850236866876216,
 8.847626889026936,
 8.844359627306163,
 8.840490247049262,
 8.835782139860491,
 8.829939152415163,
 8.822626449899364,
 8.81353334099176,
 8.802497566521641,
 8.78968581247753,
 8.775761513687684,
 8.761894876014257,
 8.749484207248758,
 8.739652786876926,
 8.732829294446528,
 8.728701342874734,
 8.726516894321632,
 8.725462334349842,
 8.724898328829665,
 8.724420912273475,
 2.7175498634087063,
 2.726240405106114,
 2.732838317383696,
 2.7412950754400067,
 2.7515588792385413,
 2.76322351798669,
 2.775652021082467,
 2.78813404829316,
 2.8000439348762525,
 2.810934784283117,
 2.8205571260602205,
 2.828827924865634,
 2.835781417141745,
 2.8415222627489576,
 2.846189415068738,
 2.8499317175851377,
 2.8528931426221473,
 2.8552049292850947,
 2.856982227516494,
 2.8583234666374087,
 4.928825858142963,
 4.925558540589685,
 4.92168909263614,
 4.916980898698522,
 4.9111377919954755,
 4.903824924984307,
 4.894731588655951,
 4.88369549895199,
 4.870883312966324,
 4.856958435930858,
 4.843091049851312,
 4.830679454750285,
 4.820846947954598,
 4.814022255112531,
 4.809893055350343,
 4.807707380707692,
 4.806651673883743,
 4.8060866361038395,
 4.805608315269433,
 4.583914985195261,
 4.5926037170142635,
 4.599199207049092,
 4.607652893223507,
 4.617912962186222,
 4.629573259893916,
 4.641996945788229,
 4.654473866442325,
 4.666378570227124,
 4.677264366723126,
 4.6868819583481205,
 4.695148434801459,
 4.702098106615368,
 4.707835662065029,
 4.712500047102311,
 4.716240072977186,
 4.719199664996662,
 4.721510008420578,
 4.723286198216269,
 4.72462661130183,
 4.6272411140132865,
 97.88830547367651,
 97.76811131071213,
 97.64264750004638,
 97.48849076829296,
 97.29289934741912,
 97.03815646920508,
 96.70040801738796,
 96.24908460211564,
 95.64794941770745,
 94.85932721211545,
 93.85300754712615,
 92.61961639370621,
 91.18430804646678,
 89.61246901758204,
 88.0,
 86.4495966673743,
 85.04478966798781,
 83.8346292441407,
 82.8330936085837,
 82.02828857399167,
 81.39411592957971,
 79.97037303108945,
 78.81404966860542,
 77.48068534295346,
 76.00398192913056,
 74.44699370853307,
 72.88378126925772,
 71.3838412211432,
 70.0,
 68.7635394564478,
 67.6858791329084,
 66.76370672606858,
 65.98465236684144,
 65.3319297240773,
 64.78752016935768,
 64.33407582665139,
 63.955897269042126,
 63.639318604560124,
 63.372747751616096,
 63.146525949335654,
 62.95270694582825,
 62.78481344831257]


lb_z = [-20.0]*40 + [-20.0]*40 + [55]*42
ub_z = [20.0]*40 + [20.0]*40 + [105]*42

model_constants = {
    'Rconst': 8.3147,
    'Nfeed': 20,
    'qF': 1,
    'T14s': 88.,
    'T28s': 70.,
    }


def ramp(val):
    """Smooth ramp function."""
    R = 20
    return cs.if_else(val < -R, 0, cs.if_else(val > R, val, 0.5*(val + cs.log(2*cs.cosh(val)))))


molar_coeff_keys = ['a', 'b', 'c', 'd']
molar_coeff_Methanol = [2.288, 0.2685, 512.4, 0.2453]
molar_coeff_nPropanol = [1.235, 0.27136, 536.4, 0.24]
molar_coeff = {
    'Methanol': dict(zip(molar_coeff_keys, molar_coeff_Methanol)),
    'nPropanol': dict(zip(molar_coeff_keys, molar_coeff_nPropanol))
    }

def density_kmol_m3(TempK : float, spec : str):
    a, b, c, d = (molar_coeff[spec][key] for key in ['a', 'b', 'c', 'd'])
    return a / b**(1 + (1 - TempK/c)**d)

def litre_per_kmol(TempK, x_Me):
    """Molar volume in litre/kmol."""
    vol_Me = x_Me / density_kmol_m3(TempK, 'Methanol')
    vol_nPropanol = (1 - x_Me) / density_kmol_m3(TempK, 'nPropanol')
    return 1000*(vol_Me + vol_nPropanol)

def kmol_per_sec_outflow(n_holdup, TempK, x_Me, refvol, flowwidth):
    """Tray outflow in kmol/s."""
    l_per_kmol = litre_per_kmol(TempK, x_Me)
    volumeholdup = l_per_kmol*n_holdup
    volumeoutflow = flowwidth*((1/500 * ramp(500*(volumeholdup - refvol)))**1.5)
    return volumeoutflow/l_per_kmol

enthalpy_coeff_keys = ['h1', 'h2', 'h3', 'Tc', 'Pc', 'OMEGA']
enthalpy_coeff_Methanol = [18.31, 1.713E-2, 6.399E-5, 512.6, 8.096E6, 0.557]
enthalpy_coeff_nPropanol = [31.92, 4.49E-2, 9.663E-5, 536.7, 5.166E6, 0.612]
enthalpy_coeff = {
    'Methanol': dict(zip(enthalpy_coeff_keys, enthalpy_coeff_Methanol)), 
    'nPropanol': dict(zip(enthalpy_coeff_keys, enthalpy_coeff_nPropanol))
    }

def enthalpies_pure(TempK, Pressure, spec):
    h1, h2, h3, Tc, Pc, OMEGA = (enthalpy_coeff[spec][key] for key in ['h1', 'h2', 'h3', 'Tc', 'Pc', 'OMEGA'])
    Rconst = model_constants['Rconst']
    
    TR = TempK/Tc
    PR = Pressure/Pc
    Dhv = Rconst*Tc*cs.sqrt(1 - PR/(TR**3))*(6.09648 - 1.28862*TR + 1.016*(TR**7) + OMEGA*(15.6875 - 13.4721*TR + 2.615*(TR**7)))
    hL = (h1*(TempK - 273.15) + h2*(TempK - 273.15)**2 + h3*(TempK - 273.15)**3)*4.186
    hV = hL + Dhv
    hL_T = (h1 + h2*2*(TempK - 273.15) + h3*3*(TempK - 273.15)**2)*4.186
    return hL, hV, hL_T


Antoine_coeff_keys = ['A', 'B', 'C']
Antoine_coeff_Methanol = [23.48, 3626.6, -34.29]
Antoine_coeff_nPropanol = [22.437, 3166.4, -80.15,]
Antoine_coeff = {
    'Methanol': dict(zip(Antoine_coeff_keys, Antoine_coeff_Methanol)),
    'nPropanol': dict(zip(Antoine_coeff_keys, Antoine_coeff_nPropanol))
    }

def pressure_at(TempK, spec):
    A, B, C = (Antoine_coeff[spec][key] for key in ('A', 'B', 'C'))
    return cs.exp(A - B/(C + TempK))

def pressure_T_at(TempK, spec):
    A, B, C = (Antoine_coeff[spec][key] for key in ('A', 'B', 'C'))
    return pressure_at(TempK, spec)*B/(C + TempK)/(C + TempK)

# x_Me: Ratio of Methanol in liquid, y: Ratio of Methanol in vapor
def enthalpies_mix(TempK, Pressure, x_Me, y):
    hL_Me, hV_Me, L_Me_T = enthalpies_pure(TempK, Pressure, 'Methanol')
    hL_nP, hV_nP, L_nP_T = enthalpies_pure(TempK, Pressure, 'nPropanol')
    
    hL = hL_Me*x_Me + hL_nP*(1 - x_Me)
    hV = hV_Me*y + hV_nP*(1 - y)
    
    hL_x = hL_Me - hL_nP
    hL_T = L_Me_T*x_Me + L_nP_T*(1 - x_Me)
    
    P_Me = pressure_at(TempK, 'Methanol')
    P_Me_T = pressure_T_at(TempK, 'Methanol')
    
    P_nP = pressure_at(TempK, 'nPropanol')
    P_nP_T = pressure_T_at(TempK, 'nPropanol')
    
    F_T = x_Me*P_Me_T + (1-x_Me)*P_nP_T
    F_x = P_Me - P_nP
    T_x = -F_x/F_T
    
    hL_x_total = hL_x + hL_T*T_x
    return hL, hV, hL_x_total

def setup_model():
    LVSCALE = 1.0e-5
    LRHSSCALE = 1.0e-5
    nSCALE = 1.0e-3
    
    # LVSCALE = 1.0
    # LRHSSCALE = 1.0
    # nSCALE = 1.0
    
    t = cs.MX.sym('t')
    
    x = cs.MX.sym('x', nx)
    x_Me = x[0:42]             #Methanol concentration
    n_internal = x[42:82]*nSCALE   #molar holdups
    
    z = cs.MX.sym('z', nz)
    L_alg = z[0:40]*LVSCALE         #(liquid flow)
    V_alg = z[40:80]*LVSCALE        #(vapor flow)
    TempC = z[80:122]       #(temperature)
    TempK = TempC + 273.15
    
    
    u = cs.MX.sym('u', nu)
    L_lh, Q = cs.vertsplit(u)
    
    alpharect, alphastrip, Qloss, nvolB, nvolD, nvolK, nvolM, flowwidth, Ptop, DPstrip, DPrect, TfeedC, TcondC, F_lh, xFeed, L_ref, Q_ref = (model_params[key] for key in ('alpharect', 'alphastrip', 'Qloss', 'nvolB', 'nvolD', 'nvolK', 'nvolM', 'flowwidth', 'Ptop', 'DPstrip', 'DPrect', 'TfeedC', 'TcondC', 'F_lh', 'xFeed', 'L_ref', 'Q_ref'))
    nvolM = nvolK
    DPrect = DPstrip
    
    TfeedK = TfeedC + 273.15
    TcondK = TcondC + 273.15
    
    Rconst, Nfeed, qF, T14s, T28s = (model_constants[key] for key in ('Rconst', 'Nfeed', 'qF', 'T14s', 'T28s'))
    
    
    Pressure = cs.MX.zeros(42)
    Pressure[41] = Ptop
    for i in range(41, 0, -1):
        Pressure[i-1] = Pressure[i] + (DPrect if i > Nfeed else DPstrip)
    
    P_Me = pressure_at(TempK, 'Methanol')
    P_nP = pressure_at(TempK, 'nPropanol')
    
    y = P_Me * x_Me/Pressure
    y_eff = y
    for i in range(1, Nfeed + 1):
        y_eff[i] = alphastrip * y[i] + (1 - alphastrip) * y_eff[i-1]
    for i in range(Nfeed + 1, 41):
        y_eff[i] = alpharect * y[i] + (1 - alpharect) * y_eff[i-1]
    y_eff[41] = x_Me[41]
    
    
    # Fluxes
    F = F_lh/(litre_per_kmol(TfeedK, xFeed) * 3600)
    Lc = L_lh/(litre_per_kmol(TcondK, x_Me[41]) * 3600)
    
    refvol = cs.MX.zeros(42)
    for i in range(42):
        # if i == 0: refvol[i] = nvolB
        # elif i < Nfeed: refvol[i] = nvolK
        if i == Nfeed: refvol[i] = nvolM  #Note: This is currently set equal to nvolK
        elif i < 41: refvol[i] = nvolK
        else: refvol[i] = nvolD
    
    n0_expr = refvol[0]/litre_per_kmol(TempK[0], x_Me[0])
    n41_expr = refvol[41]/litre_per_kmol(TempK[41], x_Me[41])
    n_holdup = cs.vertcat(n0_expr, n_internal, n41_expr)
    
    
    hL0, hV0, _ = enthalpies_mix(TempK[0], Pressure[0], x_Me[0], y_eff[0])
    V0_expr = (Q - Qloss)/(hV0 - hL0)

    # Reboiler outflow    
    L0_expr = 1/litre_per_kmol(TempK[0], x_Me[0]) * (litre_per_kmol(TempK[0], x_Me[1])*L_alg[0] - litre_per_kmol(TempK[0], y_eff[0])*V0_expr)
    
    # Condenser outflow
    V41_expr = 1/litre_per_kmol(TempK[41], y_eff[41]) * (litre_per_kmol(TempK[41], y_eff[40])*V_alg[40-1] - litre_per_kmol(TempK[41], x_Me[41])*Lc)
    
    
    L = cs.vertcat(L0_expr, L_alg, Lc)
    V = cs.vertcat(V0_expr, V_alg, V41_expr)
    
    
    ###Differential equations###
    
    x_rhs = cs.MX.zeros(42)
    n_rhs = cs.MX.zeros(40)
    
    # Reboiler
    Vmol0 = litre_per_kmol(TempK[0], x_Me[0])
    dVmol_x0 = litre_per_kmol(TempK[0], 1.0) - litre_per_kmol(TempK[0], 0.0)
    x_rhs[0] = (L[1]*x_Me[1] - V[0]*y_eff[0] - L[0]*x_Me[0])/(n_holdup[0]*(1 + dVmol_x0*x_Me[0]/Vmol0))
    
    # Trays 1 to 40
    for i in range(1, 41):
        x_rhs[i] = (V[i-1] * (y_eff[i-1] - x_Me[i]) + L[i+1] * (x_Me[i+1] - x_Me[i]) - V[i] * (y_eff[i] - x_Me[i])) / n_holdup[i]
        if i == Nfeed:
            x_rhs[i] += F * (xFeed - x_Me[i]) / n_holdup[i]
        
        n_rhs[i-1] = V[i-1] + L[i+1] - V[i] - L[i]
        if i == Nfeed:
            n_rhs[i-1] += F
    
    n_rhs = n_rhs/nSCALE
    # Condenser
    Vmol41 = litre_per_kmol(TempK[41], x_Me[41])
    dVmol_x41 = litre_per_kmol(TempK[41], 1.0) - litre_per_kmol(TempK[41], 0.0)
    x_rhs[41] = (V[40] * y_eff[40] - V[41] * y_eff[41] - L[41] * x_Me[41]) / (n_holdup[41] * (1 + dVmol_x41 * x_Me[41] / Vmol41))
    
    
    ###Algebraic equations###
    
    #Liquid reflux
    L_rhs = L[1:41] - kmol_per_sec_outflow(n_holdup[1:41], TempK[1:41], x_Me[1:41], refvol[1:41], flowwidth)
    L_rhs = L_rhs/LRHSSCALE
    
    kpso_args_func = cs.Function('kpso_args_func', [x, z, u], [n_holdup[1:41], TempK[1:41], x_Me[1:41], refvol[1:41], flowwidth])
    
    #Vapor flux
    hL_vec, hV_vec, hL_x_vec = enthalpies_mix(TempK, Pressure, x_Me, y_eff)
    P_Me_f = pressure_at(TfeedK, 'Methanol')
    P_nP_f = pressure_at(TfeedK, 'nPropanol')
    
    Pfeed = P_Me_f * xFeed + (1 - xFeed) * P_nP_f
    yfeed = P_Me_f * xFeed / Pfeed
    hLfeed, hVfeed, _ = enthalpies_mix(TfeedK, Pfeed, xFeed, yfeed)

    hL_41, hV_41, hL_x_41 = enthalpies_mix(TcondK, Pressure[41], x_Me[41], y_eff[41])
    hL_vec[41] = hL_41
    hV_vec[41] = hV_41
    hL_x_vec[41] = hL_x_41
    
    
    V_rhs = cs.MX.zeros(40)
    for i in range(1, 41):
        term = V[i-1] * (hV_vec[i-1] - hL_vec[i]) \
               - V[i] * (hV_vec[i] - hL_vec[i]) \
               + L[i+1] * (hL_vec[i+1] - hL_vec[i]) \
               - n_holdup[i] * hL_x_vec[i] * x_rhs[i]
        if i == Nfeed:
            term += F * (hLfeed - hL_vec[i]) + (1 - qF) * hVfeed
        V_rhs[i-1] = term
    
    ##Temperature
    T_rhs = 1 - P_Me/Pressure * x_Me - (1 - x_Me) * P_nP/Pressure
    
    ode_rhs = cs.vertcat(x_rhs, n_rhs)
    alg_rhs = cs.vertcat(L_rhs, V_rhs, T_rhs)
    quad_rhs = (TempC[14] - T14s)**2 + (TempC[28] - T28s)**2 + 0.05*(L_lh - L_ref)**2 + 0.05*(Q - Q_ref)**2
    
    L0_func = cs.Function('L0_func', [x, z, u], [L0_expr])
    V41_func = cs.Function('V41_func', [x, z, u], [V41_expr])
    
    dt = cs.MX.sym('dt', 1)
    DAE = {'x': x, 'z': z, 'p': cs.vertcat(dt,u), 'ode': dt*ode_rhs, 'alg': alg_rhs, 'quad': dt*quad_rhs}
    
    alg_rhs_func = cs.Function('alg_rhs_func', [x, z, u], [alg_rhs])
    x0_stage = cs.MX.sym('x0_stage', nx)
    z0_stage = cs.MX.sym('z0_stage', nz)
    
    
    DAE_lifted = {'t': t, 'x': x, 'z': z, 'p': cs.vertcat(dt, u, x0_stage, z0_stage), 
                  'ode': dt*ode_rhs, 
                  'alg': alg_rhs_func(x,z,u) - cs.exp(-5*t/dt)*alg_rhs_func(x0_stage, z0_stage, u), 
                  'quad': dt*quad_rhs
                  }
    
    # Create Functions
    ffcn = cs.Function('ffcn', [x, z, u], [ode_rhs])
    gfcn = cs.Function('gfcn', [x, z, u], [alg_rhs])
    
    
    
    kpso_func = cs.Function('kpso_func', [x,z,u], [kmol_per_sec_outflow(n_holdup[1:41], TempK[1:41], x_Me[1:41], refvol[1:41], flowwidth)])
    
    return DAE, DAE_lifted, ffcn, gfcn, L0_func, V41_func, kpso_func


DAE, DAE_lifted, ffcn, gfcn, L0_func, V41_func, kpso_func = setup_model()


ntC = 6
nt = ntC + 1

x_arr = [cs.DM(x_init)]
z_arr = []
for i in range(1,nt+1):
    x_arr.append(cs.MX.sym(f'X_s_{i}', nx))
    z_arr.append(cs.MX.sym(f'Z_s_{i}', nz))
    
u_arr = []
for i in range(ntC):
    u_arr.append(cs.MX.sym(f'u_{i}', nu))
u_arr.append(cs.DM([model_params['L_ref'], model_params['Q_ref']]))

x = cs.horzcat(*x_arr)
u = cs.horzcat(*u_arr)


alg_func = cs.Function('alg_func', [DAE['x'], DAE['z'], DAE['p']], [DAE['alg']])

x_start = cs.MX.sym('x_start', nx)
z_start = cs.MX.sym('z_start', nz)
u_start = cs.MX.sym('u_start', nu)

alg_sol = cs.rootfinder('alg_sol', 'newton', cs.Function('g', [z_start, x_start, u_start], [alg_func(x_start, z_start, cs.vertcat(cs.DM(1.0), u_start))]))
z_start_consist = alg_sol(z_init, x[:,0], u[:,0])
z_arr = [z_start_consist] + z_arr
z = cs.horzcat(*z_arr)


t0 = 0
tF = 36000
time_grid = np.concatenate([np.linspace(t0, tF/2, ntC + 1), np.array([tF])])
dt_grid = cs.diff(cs.DM(time_grid).T, 1, 1)

integrator_options = {
                'linear_solver': 'csparse',
                'sensitivity_method': 'simultaneous',
                'suppress_algebraic': True,
                'calc_ic': False,
                'fsens_err_con': False,
                'max_num_steps': 50000,
                'augmented_options': {'linear_solver': 'csparse'},
                'max_step_size':100,
                'abstol': 1e-3,
                'reltol': 1e-6,
                'enable_reverse': False
                }

daesol_single = cs.integrator('daesol_single', 'idas', DAE_lifted, 0.0, 1.0, integrator_options)

integrator_options.update({'calc_ic': True})
daesol_single_2 = cs.integrator('daesol_single_2', 'idas', DAE, 0.0, 1.0, integrator_options)


z_init_consist = alg_sol(z_init, x_init, u_init)

daesol_full = daesol_single_2.mapaccum(100)
grid_full = np.linspace(0, 36000, 101)
dt_grid_full = np.diff(cs.DM(grid_full).T, 1, 1)
u_full = cs.horzcat(*([u_init]*100))
out_full = daesol_full(x0 = x_init, z0 = z_init_consist, p = cs.vertcat(dt_grid_full, u_full))


plt.figure()
plt.plot(grid_full, np.concatenate([np.array([x_init[0]]), np.array(out_full['xf'][0,:]).reshape(-1)]))


daesol_grid = daesol_single_2.mapaccum(nt)
dt_grid = np.diff(cs.DM(time_grid).T, 1, 1)



daesol = daesol_single.map(nt, 'thread', 8)

out = daesol(x0 = x[:,:-1], z0 = z[:,:-1], p = cs.vertcat(dt_grid, u, x[:,:-1], z[:,:-1]))
F_Xf = out['xf']
F_Zf = out['zf']
F_qf = out['qf']
q_tf = cs.sum2(F_qf)


cblock_sizes = [nx + nu] * nt
consist_expr = cs.vec(cs.vertcat(x[:,1:], z[:,1:]) - cs.vertcat(F_Xf, F_Zf))

xopt_arr = [u_arr[0]]
x_start_arr = [u_init]
vblock_sizes = [nu]
vblock_dependencies = [False]
hessblock_sizes = [nu]

lbv_arr = [lb_u]
ubv_arr = [ub_u]

for i in range(1, ntC):
    xopt_arr += [x_arr[i], z_arr[i]]
    x_start_arr += [x_init, np.array(z_init_consist).reshape(-1)]
    
    lbv_arr += [lb_x, lb_z]
    ubv_arr += [ub_x, ub_z]
    
    vblock_sizes.append(nx + nz)
    vblock_dependencies.append(True)
    
    xopt_arr.append(u_arr[i])
    x_start_arr.append(u_init)
    lbv_arr.append(lb_u)
    ubv_arr.append(ub_u)
    
    vblock_sizes.append(nu)
    vblock_dependencies.append(False)
    hessblock_sizes.append(nx + nz + nu)
    

z_init_consist_2 = alg_sol(z_init_2, x_init_2, u_init)
xopt_arr += [x_arr[ntC], z_arr[ntC]]
x_start_arr += [x_init_2, np.array(z_init_consist_2).reshape(-1)]
lbv_arr += [lb_x, lb_z]
ubv_arr += [ub_x, ub_z]
vblock_sizes.append(nx + nz)
vblock_dependencies.append(True)
hessblock_sizes.append(nx + nz)

vblock_sizes.append(0)
vblock_dependencies.append(False)

xopt_arr += [x_arr[nt], z_arr[nt]]
x_start_arr += [x_init_2, np.array(z_init_consist_2).reshape(-1)]
lbv_arr += [lb_x, lb_z]
ubv_arr += [ub_x, ub_z]
vblock_sizes.append(nx + nz)
vblock_dependencies.append(True)
hessblock_sizes.append(nx + nz)


lbc_arr = [0]*consist_expr.numel()
ubc_arr = [0]*consist_expr.numel()
constr_arr = []
for i in range(0, nt):
    constr_arr.append(L0_func(x_arr[i], z_arr[i], u_arr[i]))
    lbc_arr.append(0.)
    ubc_arr.append(np.inf)
    
    constr_arr.append(V41_func(x_arr[i], z_arr[i], u_arr[i]))
    lbc_arr.append(0.)
    ubc_arr.append(np.inf)

cblock_sizes.append(2*nt)


x_start = np.concatenate(x_start_arr)
xopt = cs.vertcat(*xopt_arr)

lb_var = np.concatenate(lbv_arr)
ub_var = np.concatenate(ubv_arr)
lb_con = np.array(lbc_arr)
ub_con = np.array(ubc_arr)


constr_expr = cs.vertcat(consist_expr, *constr_arr)
obj_expr = q_tf*1e-3


f = cs.Function('f', [xopt], [obj_expr])
g = cs.Function('g', [xopt], [constr_expr])


grad_f_expr = cs.jacobian(obj_expr, xopt)
grad_f = cs.Function('grad_f', [xopt], [grad_f_expr])


jac_g_expr = cs.jacobian(constr_expr, xopt)
jac_g = cs.Function('jac_g', [xopt], [jac_g_expr])

g_u_expr = cs.jacobian(constr_expr, cs.vertcat(u_arr[0], x_arr[1], z_arr[1], u_arr[1]))
g_u = cs.Function('g_u', [xopt], [g_u_expr])

f_u_expr = cs.jacobian(obj_expr, cs.vertcat(u_arr[0], x_arr[1]))
f_u = cs.Function('f_u', [xopt], [f_u_expr])

NLP = {'x': xopt, 'f': obj_expr, 'g': constr_expr}
S = cs.nlpsol('S', 'ipopt', NLP, {'ipopt':{'hessian_approximation': 'limited-memory', 'max_iter': 40}})
out = S(x0 = x_start, lbx = lb_var, ubx = ub_var, lbg = lb_con, ubg = ub_con)


xi_sol = out['x']
L_sol = xi_sol[0:(82+122+2)*ntC:(82+122+2)]
Q_sol = xi_sol[1:(82+122+2)*ntC:(82+122+2)]
u_sol = cs.vertcat(L_sol.T, Q_sol.T)

x_sol_arr = [x_init]
start = 2
for i in range(ntC):
    x_sol_arr.append(xi_sol[start:start+82])
    start += 82 + 122 + 2
start -= 2
x_sol_arr.append(xi_sol[start:start+82])
x_sol = cs.horzcat(*x_sol_arr)

z_sol_arr = [alg_sol(z_init, x_init, u_sol[:,0])]
start = 2 + 82
for i in range(ntC):
    z_sol_arr.append(xi_sol[start:start+122])
    start += 82 + 122 + 2
start -= 2
z_sol_arr.append(xi_sol[start:start+122])
z_sol = cs.horzcat(*z_sol_arr)




# import blockSQP2

# jac_g0 = jac_g(x_start)
# prob = blockSQP2.Problem(xopt.numel(), constr_expr.numel())
# prob.f = lambda x: float(f(x))
# prob.grad_f = lambda x: np.array(grad_f(x)).reshape(-1)

# prob.g = lambda x: np.array(g(x)).reshape(-1)
# prob.make_sparse(jac_g0.nnz, jac_g0.row(), jac_g0.colind())
# prob.jac_g_nz = lambda x: np.array(jac_g(x).nz[:]).reshape(-1)

# prob.set_bounds(lb_var, ub_var, lb_con, ub_con)

# prob.blockIdx = np.cumsum(np.array([0] + hessblock_sizes))

# prob.x_start = x_start
# prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)

# stats = blockSQP2.Stats("./")

# vblocks = [blockSQP2.vblock(sz, dp) for sz, dp in zip(vblock_sizes, vblock_dependencies)]
# cblocks = [blockSQP2.cblock(sz) for sz in cblock_sizes]
# hsizes = hess_block_sizes
# targets = [blockSQP2.condensing_target()]


# opts = blockSQP2.Options(
    
#     )

# optimizer = blockSQP2.Solver(prob, opts, stats)
# optimizer.init()
# ret = optimizer.run(10)






