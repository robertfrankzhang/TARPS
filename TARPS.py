
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import random
from OfflineEnvironment import *
from Timer import *

class TARPS(BaseEstimator):
    def __init__(self,learningIterations = 1000, N=1000,p_spec=0.5,discreteAttributeLimit=10,specifiedAttributes = [],
                 discretePhenotypeLimit=10,nu=5,chi=0.8,upsilon=0.04,theta_GA=25,theta_del=20,beta=0.2,delta=0.1,
                 init_fit=0.01,fitnessReduction=0.1,theta_sel=0.5,randomSeed="none"):
        '''
        :param learningIterations:      Must be nonnegative integer. The number of training cycles to run.
        :param trackingFrequency:       Must be nonnegative integer. Relevant only if evalWhileFit param is true. Conducts accuracy approximations and population measurements every trackingFrequency iterations.
                                        If param == 0, tracking done once every epoch.
        :param N:                       Must be nonnegative integer. Maximum micro classifier population size (sum of classifier numerosities).
        :param p_spec:                  Must be float from 0 - 1. Probability of specifying an attribute during the covering procedure. Advised: larger amounts of attributes => lower p_spec values
        :param discreteAttributeLimit:  Must be nonnegative integer OR "c" OR "d". Multipurpose param. If it is a nonnegative integer, discreteAttributeLimit determines the threshold that determines
                                        if an attribute will be treated as a continuous or discrete attribute. For example, if discreteAttributeLimit == 10, if an attribute has more than 10 unique
                                        values in the dataset, the attribute will be continuous. If the attribute has 10 or less unique values, it will be discrete. Alternatively,
                                        discreteAttributeLimit can take the value of "c" or "d". See next param for this.
        :param specifiedAttributes:     Must be an ndarray type of nonnegative integer attributeIndices (zero indexed).
                                        If "c", attributes specified by index in this param will be continuous and the rest will be discrete. If "d", attributes specified by index in this
                                        param will be discrete and the rest will be continuous.
                                        If this value is given, and discreteAttributeLimit is not "c" or "d", discreteAttributeLimit overrides this specification
        :param discretePhenotypeLimit:  Must be nonnegative integer OR "c" OR "d". Works similarly to discreteAttributeLimit. Multipurpose param. If it is a nonnegative integer, this param determines the
                                        continuous/discrete threshold for the phenotype. If it is "c" or "d", the phenotype is explicitly defined as continuous or discrete.
        :param nu:                      (v) Must be a float. Power parameter used to determine the importance of high accuracy when calculating fitness. (typically set to 5, recommended setting of 1 in noisy data)
        :param chi:                     (X) Must be float from 0 - 1. The probability of applying crossover in the GA. (typically set to 0.5-1.0)
        :param upsilon:                 (u) Must be float from 0 - 1. The probability of mutating an allele within an offspring.(typically set to 0.01-0.05)
        :param theta_GA:                Must be nonnegative float. The GA threshold. The GA is applied in a set when the average time (# of iterations) since the last GA in the correct set is greater than theta_GA.
        :param theta_del:               Must be a nonnegative integer. The deletion experience threshold; The calculation of the deletion probability changes once this threshold is passed.
        :param beta:                    Must be float. Learning parameter; Used in calculating average correct set size
        :param delta:                   Must be float. Deletion parameter; Used in determining deletion vote calculation.
        :param init_fit:                Must be float. The initial fitness for a new classifier. (typically very small, approaching but not equal to zero)
        :param fitnessReduction:        Must be float. Initial fitness reduction in GA offspring rules.
        :param theta_sel:               Must be float from 0 - 1. The fraction of the correct set to be included in tournament selection.
        :param randomSeed:              Must be an integer or "none". Set a constant random seed value to some integer (in order to obtain reproducible results). Put 'none' if none (for pseudo-random algorithm runs).
        '''

        '''
        Parameter Validity Checking
        Checks all parameters for valid values
        '''
        # learningIterations
        if not self.checkIsInt(learningIterations):
            raise Exception("learningIterations param must be nonnegative integer")

        if learningIterations < 0:
            raise Exception("learningIterations param must be nonnegative integer")

        # N
        if not self.checkIsInt(N):
            raise Exception("N param must be nonnegative integer")

        if N < 0:
            raise Exception("N param must be nonnegative integer")

        # p_spec
        if not self.checkIsFloat(p_spec):
            raise Exception("p_spec param must be float from 0 - 1")

        if p_spec < 0 or p_spec > 1:
            raise Exception("p_spec param must be float from 0 - 1")

        # discreteAttributeLimit
        if discreteAttributeLimit != "c" and discreteAttributeLimit != "d":
            try:
                dpl = int(discreteAttributeLimit)
                if not self.checkIsInt(discreteAttributeLimit):
                    raise Exception("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'")
                if dpl < 0:
                    raise Exception("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'")
            except:
                raise Exception("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'")

        # specifiedAttributes
        if not (isinstance(specifiedAttributes, list)):
            raise Exception("specifiedAttributes param must be ndarray")

        for spAttr in specifiedAttributes:
            if not self.checkIsInt(spAttr):
                raise Exception("All specifiedAttributes elements param must be nonnegative integers")
            if int(spAttr) < 0:
                raise Exception("All specifiedAttributes elements param must be nonnegative integers")

        # discretePhenotypeLimit
        if discretePhenotypeLimit != "c" and discretePhenotypeLimit != "d":
            try:
                dpl = int(discretePhenotypeLimit)
                if not self.checkIsInt(discretePhenotypeLimit):
                    raise Exception("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'")
                if dpl < 0:
                    raise Exception("discretePhenotypeLimit param must be nonnegative integer or 'c' or 'd'")
            except:
                raise Exception("discretePhenotypeLimit param must be nonnegative integer or 'c' or 'd'")

        # nu
        if not self.checkIsFloat(nu):
            raise Exception("nu param must be float")

        # chi
        if not self.checkIsFloat(chi):
            raise Exception("chi param must be float from 0 - 1")

        if chi < 0 or chi > 1:
            raise Exception("chi param must be float from 0 - 1")

        # upsilon
        if not self.checkIsFloat(upsilon):
            raise Exception("upsilon param must be float from 0 - 1")

        if upsilon < 0 or upsilon > 1:
            raise Exception("upsilon param must be float from 0 - 1")

        # theta_GA
        if not self.checkIsFloat(theta_GA):
            raise Exception("theta_GA param must be nonnegative float")

        if theta_GA < 0:
            raise Exception("theta_GA param must be nonnegative float")

        # theta_del
        if not self.checkIsInt(theta_del):
            raise Exception("theta_del param must be nonnegative integer")

        if theta_del < 0:
            raise Exception("theta_del param must be nonnegative integer")

        # beta
        if not self.checkIsFloat(beta):
            raise Exception("beta param must be float")

        # delta
        if not self.checkIsFloat(delta):
            raise Exception("delta param must be float")

        # init_fit
        if not self.checkIsFloat(init_fit):
            raise Exception("init_fit param must be float")

        # fitnessReduction
        if not self.checkIsFloat(fitnessReduction):
            raise Exception("fitnessReduction param must be float")

        # theta_sel
        if not self.checkIsFloat(theta_sel):
            raise Exception("theta_sel param must be float from 0 - 1")

        if theta_sel < 0 or theta_sel > 1:
            raise Exception("theta_sel param must be float from 0 - 1")

        # randomSeed
        if randomSeed != "none":
            try:
                if not self.checkIsInt(randomSeed):
                    raise Exception("randomSeed param must be integer or 'none'")
                random.seed(int(randomSeed))
                np.random.seed(int(randomSeed))
            except:
                raise Exception("randomSeed param must be integer or 'none'")

        '''
        Set params
        '''
        self.learningIterations = learningIterations
        self.N = N
        self.p_spec = p_spec
        self.discreteAttributeLimit = discreteAttributeLimit
        self.discretePhenotypeLimit = discretePhenotypeLimit
        self.specifiedAttributes = specifiedAttributes
        self.nu = nu
        self.chi = chi
        self.upsilon = upsilon
        self.theta_GA = theta_GA
        self.theta_del = theta_del
        self.beta = beta
        self.delta = delta
        self.init_fit = init_fit
        self.fitnessReduction = fitnessReduction
        self.theta_sel = theta_sel
        self.randomSeed = randomSeed

    def checkIsInt(self, num):
        try:
            n = float(num)
            if num - int(num) == 0:
                return True
            else:
                return False
        except:
            return False

    def checkIsFloat(self, num):
        try:
            n = float(num)
            return True
        except:
            return False

    def fit(self,X,y):
        '''
        Unlike most Scikit-learn packages, X must be in a particular format in order to be correctly analyzed. Patient instances
        must be separated by a row of all NaN types. All NaN type rows mark delineations between patient instances. If
        these rows don't exist, TARPS will treat the entire dataset as a single patient instance.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
        y: array-like {n_samples}
            Training labels. ALL INSTANCE PHENOTYPES MUST BE NUMERIC. IT IS ASSUMED TO BE AS SUCH. However, NaN in
            this column is acceptable at locations where there is a NaN line break

        Returns
        __________
        self
        '''

        # Check if X and Y are numeric
        try:
            for instance in X:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
            for value in y:
                if not (np.isnan(value)):
                    float(value)

        except:
            raise Exception("X and y must be fully numeric")

        self.env = OfflineEnvironment(self,X,y)

        if not self.env.formatData.discretePhenotype:
            raise Exception("eLCS works best with classification problems. While we have the infrastructure to support continuous phenotypes, we have disabled it for this version.")

        self.timer = Timer()
        self.population = None #Initialize Population
        self.explorIter = 0

        while self.explorIter < self.learningIterations:
            state_phenotype = self.env.getTrainInstance()
            self.runIteration(state_phenotype, self.explorIter)
            self.explorIter += 1
            self.env.newInstance()

        return self

    def runIteration(self, state_phenotype, exploreIter):
        print("iteration")
