import numpy as np
import quadpy
import warnings


class CubatureRules:
    """Provides cubature rules for line, triangle, and tetrahedral

    Attributes:
        p (int): maximum degree of polynomial for which the cubature rule approximates exactly
        quad_type (str):  In 1D -- 'LGL', 'LG'

    Methods:
        quad_line_volume()
        quad_line_face()
    """

    @staticmethod
    def quad_line_volume(p, quad_type, n=0):
        """Returns the nodal location and weights associated with quadrature rules in 1D
            **kwargs: n (int):     -- number of degrees of freedom"""

        if n == 0:
            n = p+1

        if quad_type == 'LG':
            if n != p + 1:
                warnings.warn('Mismatch between degree of operator and number of degrees of freedom: n = p+1.')
            quadrule = quadpy.line_segment.gauss_legendre(n)
            xq = quadrule.points.reshape((n, 1))
            wq = quadrule.weights.reshape((n, 1))

        elif quad_type == 'LGL':
            if n != p + 1:
                warnings.warn('Mismatch between degree of operator and number of degrees of freedom: n = p+1.')
            quadrule = quadpy.line_segment.gauss_lobatto(n)
            xq = quadrule.points.reshape((n, 1))
            wq = quadrule.weights.reshape((n, 1))

        else:
            raise NotImplementedError
        return xq, wq

    @staticmethod
    def quad_line_face(p, quad_type):
        xqf = np.array([-1, 1])
        wqf = 1.0
        return xqf, wqf

    @staticmethod
    def cub_tri_volume(p, sbp_family, p_cub=0):
        """A module to obtain the cubature points and weights for SBP-Gamma, SBP-Omega and SBP-DiagE operators
        Args:
            p(int): degree of the cubature rule
            sbp_family (str): the type of cubature rule, "SBP-Gamma", "SBP-Omega", or "SBP-DiagE"
            p_cub (int, optional): the degree of the cubature rule which is set to 2p-1

        Returns:
            r(array): the x coordinates of the cubature points on the reference right triangle
            s(array): the y coordinates of the cubature points on the reference right triangle
            w(array): the weight of the cubature rule at the cubature points

        """
        # order of the cubature rule
        if p_cub == 0:
            p_cub = 2*p - 1     # assume 2p-1 if not provided

        # the reference triangle on which the cubature rules are given
        cub_vert_BigTriangle = np.array([[-1, -1], [1, -1], [-1, 1]])
        cub_vert_SmallTriangle = np.array([[0, 0], [1, 0], [0, 1]])
        cub_vert = cub_vert_BigTriangle     # assume cubature rule is on the big triangle unless specified

        sbp_family = str.lower(sbp_family)
        if sbp_family == "gamma":
            if p == 1:
                if p_cub == 1:
                    r = np.array([-1.0,  1.0, -1.0])
                    s = np.array([-1.0, -1.0,  1.0])
                    w = np.array([0.6666666666666666,  0.6666666666666666,  0.6666666666666666])
                elif p_cub == 2:
                    r = np.array([0.333333333333333, 0, 1, 0])
                    s = np.array([0.333333333333333, 0, 0, 1])
                    w = np.array([0.375, 0.04166666666666666, 0.04166666666666666, 0.04166666666666666])
                    cub_vert = cub_vert_SmallTriangle
                else:
                    raise ValueError("Only cubature rules of degree p_cub=1 and p_cub=2 are available.")

            elif p == 2:
                if p_cub == 3:
                    r = np.array([-1.0, 1.0, -1.0, 0.0, 0.0, -1.0, -0.3333333333333333])
                    s = np.array([-1.0, -1.0,  1.0, -1.0, 0.0,  0.0, -0.3333333333333333])
                    w = np.array([0.09999999999999999,  0.09999999999999999,  0.09999999999999999,  0.26666666666666666, 0.26666666666666666,  0.26666666666666666,  0.9000000000000002])
                elif p_cub == 4:
                    r = np.array([0.188580484696445, 0.6228390306071099, 0.188580484696445, 0, 1, 0, 0.5, 0, 0.5])
                    s = np.array([0.188580484696445, 0.188580484696445, 0.6228390306071099, 0, 0, 1, 0.5, 0.5, 0])
                    w = np.array([0.1254088495595664, 0.1254088495595664, 0.1254088495595664, 0.01027006767296668, 0.01027006767296668, 0.01027006767296668, 0.03098774943413357, 0.03098774943413357, 0.03098774943413357])
                    cub_vert = cub_vert_SmallTriangle
                else:
                    raise ValueError("Only cubature rules of degree p_cub=3 and p_cub=4 are available.")

            elif p == 3:
                if p_cub == 5:
                    r = np.array([-1.0,  1.0, -1.0, -0.4130608881819197, 0.4130608881819197,  0.4130608881819197, -0.4130608881819197, -1.0, -1.0,  -0.5853096486728182,  0.17061929734563638, -0.5853096486728182])
                    s = np.array([-1.0, -1.0,  1.0, -1.0, -1.0, -0.4130608881819197,  0.4130608881819197,  0.4130608881819197, -0.4130608881819197,  -0.5853096486728182, -0.5853096486728182,  0.17061929734563636])
                    w = np.array([0.02974582604964118,  0.02974582604964118,  0.02974582604964118,  0.09768336246810204, 0.09768336246810204,  0.09768336246810204,  0.09768336246810204,  0.09768336246810204, 0.09768336246810204,   0.4415541156808217,  0.4415541156808217,  0.4415541156808217])
                else:
                    raise ValueError("Only cubature rules of degree p_cub=5 is available.")

            elif p == 4:
                if p_cub == 7:
                    r = np.array([-1.0,  1.0, -1.0, -0.5773502691896257,  0.0,  0.5773502691896257, 0.5773502691896257, 0.0, -0.5773502691896257, -1.0, -1.0, -1.0, -0.7384168123405102, -0.1504720765483788,  0.4768336246810203, -0.1504720765483788, -0.7384168123405102, -0.6990558469032424])
                    s = np.array([-1.0, -1.0,  1.0, -1.0, -1.0, -1.0, -0.5773502691896257, 0.0, 0.5773502691896257, 0.5773502691896257,  0.0, -0.5773502691896257, -0.7384168123405102, -0.6990558469032424, -0.7384168123405102, -0.15047207654837885,  0.47683362468102025, -0.15047207654837885])
                    w = np.array([0.012698412698412695, 0.012698412698412695, 0.012698412698412695, 0.04285714285714284, 0.05079365079365077,  0.04285714285714284,  0.04285714285714284, 0.05079365079365077,  0.04285714285714284,  0.04285714285714284,  0.05079365079365077,  0.04285714285714284,  0.2023354595827503,  0.3151248578775673,  0.2023354595827503,  0.3151248578775673,  0.2023354595827503,  0.3151248578775673])
                else:
                    raise ValueError("Only cubature rules of degree p_cub=7 is available.")

            elif p >= 5:
                raise ValueError("Only cubature rules of degree up to p=4 are available for the SBP-Gamma family.")

        elif sbp_family == "omega":
            if p==1:
                if p_cub==1 or p_cub==2:
                    r = np.array([-0.6666666666666667, 0.3333333333333335, -0.6666666666666667])
                    s = np.array([-0.6666666666666667, -0.6666666666666667, 0.3333333333333334])
                    w = np.array([0.6666666666666666, 0.6666666666666666, 0.6666666666666666])
                else:
                    raise ValueError("Only cubature rules of degrees p_cub=1 and p_cub=2 are available.")
            elif p==2:
                if p_cub==3 or p_cub==4:
                    r = np.array([-0.8168475729804585,  0.633695145960917 , -0.8168475729804585, -0.10810301816807022, -0.10810301816807022, -0.7837939636638596])
                    s = np.array([-0.8168475729804585, -0.8168475729804585,  0.633695145960917 , -0.7837939636638596 , -0.10810301816807022, -0.10810301816807022])
                    w = np.array([0.2199034873106437,  0.2199034873106437,  0.2199034873106437, 0.44676317935602283 ,  0.44676317935602283,  0.44676317935602283])
                elif p_cub==5:
                    r = np.array([0.3333333333333333, 0.1012865073234563, 0.7974269853530873, 0.1012865073234563, 0.4701420641051151, 0.05971587178976982, 0.4701420641051151])
                    s = np.array([0.3333333333333333, 0.1012865073234563, 0.1012865073234563, 0.7974269853530873, 0.4701420641051151, 0.4701420641051151, 0.05971587178976982])
                    w = np.array([0.1125, 0.06296959027241357, 0.06296959027241357, 0.06296959027241357, 0.0661970763942531, 0.0661970763942531, 0.0661970763942531])
                    cub_vert = cub_vert_SmallTriangle
                else:
                    raise ValueError("Only cubature rules of degrees p_cub=3, p_cub=4, and p_cub=5 are available.")
            elif p==3:
                if p_cub==5:
                    r = np.array([-0.8613766937233731 ,  0.7227533874467464 , -0.8613766937233732 , -0.3773557414367168 ,  0.23519630088171353,  0.2351963008817135 , -0.3773557414367168 , -0.8578405594449967 , -0.8578405594449967 , -0.3333333333333333])
                    s = np.array([-0.8613766937233733 , -0.8613766937233733 ,  0.7227533874467464 , -0.8578405594449967 , -0.8578405594449967 , -0.3773557414367168 ,  0.2351963008817135 ,  0.2351963008817135 , -0.3773557414367168 , -0.3333333333333333 ])
                    w = np.array([0.11550472674301035,  0.11550472674301035,  0.11550472674301035,  0.20924480696331949,  0.20924480696331949,  0.20924480696331949,  0.20924480696331949,  0.20924480696331949,  0.20924480696331949,  0.39801697799105223])
                else:
                    raise ValueError("Only cubature rules of degree p_cub=5 is available.")
            elif p==4:
                if p_cub==7:
                    r = np.array([-0.9156687711811358  ,  0.8313375423622715  , -0.9156687711811358  , -0.5768758827238152 , -0.05141062176497946,  0.48091319998088583,  0.4809131999808858 , -0.05141062176497946, -0.5768758827238151 , -0.9040373172570707 , -0.8971787564700411 , -0.9040373172570706 , -0.5158280524810427,  0.031656104962085374, -0.5158280524810427])
                    s = np.array([-0.9156687711811358  , -0.9156687711811358  ,  0.8313375423622715  , -0.9040373172570706 , -0.8971787564700411 , -0.9040373172570706 , -0.5768758827238152 , -0.05141062176497946,  0.4809131999808858 ,  0.4809131999808858 , -0.05141062176497946, -0.5768758827238152 , -0.5158280524810426, -0.5158280524810426  ,  0.031656104962085374])
                    w = np.array([0.045386157905236924,  0.045386157905236924,  0.045386157905236924,  0.11055758694624952,  0.145828414950907  ,  0.11055758694624952,  0.11055758694624952,  0.145828414950907  ,  0.11055758694624952,  0.11055758694624952,  0.145828414950907  ,  0.11055758694624952,  0.2543369199180238,  0.2543369199180238  ,  0.2543369199180238])
                else:
                    raise ValueError("Only cubature rules of degree p_cub=4 is available.")
            elif p >= 5:
                raise ValueError("Only cubature rules of degree up to p=4 are available for the SBP-Omega family.")

        elif sbp_family == "diage":
            if p==1:
                if p_cub==1:
                    r = np.array([-0.577350269189626,  0.577350269189626,  0.577350269189626,  -0.577350269189626,  -1.000000000000000,   -1.000000000000000])
                    s = np.array([-1.000000000000000, -1.000000000000000,  -0.577350269189626,  0.577350269189626,  0.577350269189626,   -0.577350269189626])
                    w = np.array([0.333333333333333,   0.333333333333333,  0.333333333333333,  0.333333333333333,  0.333333333333333,  0.333333333333333])
                elif p_cub==2:
                    r = np.array([0.3333333333333333, 0.7886751345948129, 0, 0.2113248654051871, 0.2113248654051871, 0, 0.7886751345948129])
                    s = np.array([0.3333333333333333, 0.2113248654051871, 0.7886751345948129, 0, 0.7886751345948129, 0.2113248654051871, 0])
                    w = np.array([0.25, 0.04166666666666666, 0.04166666666666666, 0.04166666666666666, 0.04166666666666666, 0.04166666666666666, 0.04166666666666666])
                    cub_vert = cub_vert_SmallTriangle
                else:
                    raise ValueError("Only cubature rules of degree p_cub=1 and p_cub=2 are available.")
            elif p==2:
                if p_cub==3:
                    r = np.array([-0.774596669241483,  0, 0.774596669241483, 0.774596669241483, 0, -0.774596669241483, -1.000000000000000, -1.000000000000000, -1.000000000000000, -0.333333333333333])
                    s = np.array([-1.000000000000000,  -1.000000000000000, -1.000000000000000, -0.774596669241483, 0, 0.774596669241483, 0.774596669241483, 0, -0.774596669241483, -0.333333333333333])
                    w = np.array([0.083333333333333,  0.200000000000000, 0.083333333333333, 0.083333333333333, 0.200000000000000, 0.083333333333333,  0.083333333333333, 0.200000000000000, 0.083333333333333, 0.900000000000000])
                elif p_cub==4:
                    r = np.array([0.2046806415707621, 0.5906387168584759, 0.2046806415707621, 0.8872983346207417, 0, 0.1127016653792583, 0.1127016653792583, 0, 0.8872983346207417, 0.5, 0, 0.5])
                    s = np.array([0.2046806415707621, 0.2046806415707621, 0.5906387168584759, 0.1127016653792583, 0.8872983346207417, 0, 0.8872983346207417, 0.1127016653792583, 0, 0.5, 0.5, 0])
                    w = np.array([0.1122592271822205, 0.1122592271822205, 0.1122592271822205, 0.01260251572603941, 0.01260251572603941, 0.01260251572603941, 0.01260251572603941, 0.01260251572603941, 0.01260251572603941, 0.02920240803236731, 0.02920240803236731, 0.02920240803236731])
                    cub_vert = cub_vert_SmallTriangle
                else:
                    raise ValueError("Only cubature rules of degree p_cub=3 and p_cub=4 are available.")
            elif p==3:
                if p_cub==5:
                    r = np.array([-0.861136311594053 ,-0.339981043584856 , 0.339981043584856 , 0.861136311594053 , 0.861136311594053 , 0.339981043584856 ,-0.339981043584856 ,-0.861136311594053 ,-1                 ,-1                 ,-1                 ,-1                 ,-0.542461986333868, 0.168314227951314, 0.168314227951314,-0.542461986333868,-0.625852241617446,-0.625852241617446  ])
                    s = np.array([-1                 ,-1                 ,-1                 ,-1                 ,-0.861136311594053 ,-0.339981043584856 , 0.339981043584856 , 0.861136311594053 , 0.861136311594053 , 0.339981043584856 ,-0.339981043584856 ,-0.861136311594053 ,-0.625852241617446,-0.625852241617446,-0.542461986333868, 0.168314227951314, 0.168314227951314,-0.542461986333868])
                    w = np.array([0.0301980297451310, 0.0809130813659800, 0.0809130813659800, 0.0301980297451310, 0.0301980297451310, 0.0809130813659800, 0.0809130813659800, 0.0301980297451310, 0.0301980297451310, 0.0809130813659800, 0.0809130813659800, 0.0301980297451310, 0.222222222222222, 0.222222222222222, 0.222222222222222, 0.222222222222222, 0.222222222222222, 0.222222222222222])
                else:
                    raise ValueError("Only cubature rules of degree p_cub=5 is available.")
            elif p==4:
                if p_cub==7:
                    r = np.array([-0.906179845938664 ,-0.538469310105682 , 0                 , 0.538469310105682 , 0.906179845938664 , 0.906179845938664 , 0.538469310105682 ,0                 ,-0.538469310105682 ,-0.906179845938664 ,-1                 ,-1                 ,-1                 ,-1                 ,-1                 ,-0.721132537169093,-0.123152095118363, 0.442265074338186,-0.123152095118363,-0.721132537169093,-0.753695809763274,-0.333333333333333])
                    s = np.array([-1                 ,-1                 ,-1                 ,-1                 ,-1                 ,-0.906179845938664 ,-0.538469310105682 ,0                 , 0.538469310105682 , 0.906179845938664 , 0.906179845938664 , 0.538469310105682 , 0                 ,-0.538469310105682 ,-0.906179845938664 ,-0.721132537169093,-0.753695809763274,-0.721132537169093,-0.123152095118363, 0.442265074338186,-0.123152095118363,-0.333333333333333])
                    w = np.array([0.0132026301620030, 0.0410609193608580, 0.0370741696679000, 0.0410609193608580, 0.0132026301620030, 0.0132026301620030, 0.0410609193608580,0.0370741696679000, 0.0410609193608580, 0.0132026301620030, 0.0132026301620030, 0.0410609193608580, 0.0370741696679000, 0.0410609193608580, 0.0132026301620030, 0.210858659241689, 0.249473464579547, 0.210858659241689, 0.249473464579547, 0.210858659241689, 0.249473464579547, 0.182199822395427])
                else:
                    raise ValueError("Only cubature rules of degree p_cub=7 is available.")
            elif p >= 5:
                raise ValueError("Only cubature rules of degree up to p=4 are available for the SBP-DiagE family.")

        r = r.reshape(r.shape[0], 1)
        s = s.reshape(s.shape[0], 1)
        w = w.reshape(w.shape[0], 1)

        return {'r': r, 's': s, 'w': w, 'cub_vert': cub_vert}


# quad = CubatureRules(5, 'LG')
# xp, xw = quad.quad_line_volume()
# print(xp)
# print(xw)

# cub = CubatureRules.cub_tri_volume(1, "SBP-gamma")
# print(cub['s'])