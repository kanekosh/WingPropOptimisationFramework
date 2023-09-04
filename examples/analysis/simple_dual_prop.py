# --- Built-ins ---

# --- Internal ---
from src.base import ParamInfo, WingPropInfo, WingInfo, PropInfo, AirfoilInfo
from src.integration.coupled_groups_analysis import WingSlipstreamProp

# --- External ---
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots

spanwise_discretisation_propeller_BEM = 20
spanwise_section = 0.12/(spanwise_discretisation_propeller_BEM+1)
rot_rate = 2100
prop_twist = np.array(np.rad2deg(
                            np.cos(
                                    np.linspace(0.15, .6, spanwise_discretisation_propeller_BEM+1)*0.5*np.pi
                                    )
                                ),
                        order='F')

ref_point = np.array([0.0,
                        0.01,
                        0.0], order='F')

parameters = ParamInfo(vinf=35.,
                       wing_aoa=2.,
                       mach_number=0.2,
                       reynolds_number=1e6,
                       speed_of_sound=333.4)

wing = WingInfo(label='SampleWing',
                span=10.,
                chord=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F'),
                twist=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*0.1,
                empty_weight=10.
                )

prop1 = PropInfo(label='Prop1',
                 prop_location=-2.,
                 nr_blades=4,
                 rot_rate=rot_rate,
                 chord=np.ones(spanwise_discretisation_propeller_BEM+1,
                               order='F')*spanwise_section,
                 twist=prop_twist,
                 span=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*spanwise_section,
                 airfoils=[AirfoilInfo(label='SampleFoil',
                                       Cl_alpha=6.22,
                                       alpha_L0=0.,
                                       alpha_0=0.24)]*(spanwise_discretisation_propeller_BEM+1),
                 ref_point=ref_point
                 )

prop2 = PropInfo(label='Prop2',
                 prop_location=2.,
                 nr_blades=4,
                 rot_rate=rot_rate,
                 chord=np.ones(spanwise_discretisation_propeller_BEM+1,
                               order='F')*spanwise_section,
                 twist=prop_twist,
                 span=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*spanwise_section,
                 airfoils=[AirfoilInfo(label='SampleFoil',
                                       Cl_alpha=6.22,
                                       alpha_L0=0.,
                                       alpha_0=0.24)]*(spanwise_discretisation_propeller_BEM+1),
                 ref_point=ref_point
                 )

wingpropinfo = WingPropInfo(spanwise_discretisation_wing=60,
                            spanwise_discretisation_propeller=21,
                            spanwise_discretisation_propeller_BEM=spanwise_discretisation_propeller_BEM,
                            propeller=[prop1, prop2],
                            wing=wing,
                            parameters=parameters
                            )


class WingSlipstreamPropAnalysis(om.Group):
    def initialize(self):
        self.options.declare('WingPropInfo', default=WingPropInfo)

    def setup(self):
        self.add_subsystem('PropellerSlipstreamWingModel',
                           subsys=WingSlipstreamProp(WingPropInfo=wingpropinfo))

    def configure(self):
        # Empty because we do analysis
        ...


if __name__ == '__main__':
    prob = om.Problem()
    prob.model = WingSlipstreamPropAnalysis(WingPropInfo=wingpropinfo)

    prob.setup()
    prob.run_model()
    Cl_corr = prob['PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.aero_funcs.liftcoeff.Cl']

    prob.model.options['WingPropInfo'].NO_CORRECTION = True
    prob.setup()
    prob.run_model()
    
    Cl_nocorr = prob['PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.aero_funcs.liftcoeff.Cl']
    V_distr = prob['PropellerSlipstreamWingModel.RETHORST.velocity_distribution']

    plt.style.use(niceplots.get_style())
    _, ax = plt.subplots(2, figsize=(10, 7))

    spanwise = np.linspace(-wingpropinfo.wing.span/2,
                           wingpropinfo.wing.span/2,
                           len(Cl_nocorr))
    ax[0].plot(spanwise, Cl_nocorr, label='Lift coefficient, no correction')
    ax[0].plot(spanwise, Cl_corr, label='Lift coefficient, with correction')

    ax[0].set_xlabel(r'Spanwise location $y$')
    ax[0].set_ylabel(r'$C_L\cdot c$')
    ax[0].legend()
    niceplots.adjust_spines(ax[0], outward=True)
    
    ax[1].plot(spanwise, V_distr, label='Velocity distribution')

    ax[1].set_xlabel(r'Spanwise location $y$')
    ax[1].set_ylabel(r'$V$')
    ax[1].legend()
    niceplots.adjust_spines(ax[1], outward=True)

    plt.savefig('figure.png')
