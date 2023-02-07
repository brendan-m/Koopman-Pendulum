# Pendulum model
#
#  out:
#      model    - model struct, containing
#           .l  - string/rod length
#           .m  - bob mass
#           .mu - viscous friction
#
# Original author = Matthew J. Howard
# Converted to python = Brendan Michael

def model_pendulum(L1=1,M1=1,mu=0,G=-9.81):
# model geometry and dynamics parameters
    model = {
        'L1':L1,   # Link length
        'M1':M1,   # Link mass
        'mu':mu,   # Viscous friction
        'G':G, # Gravity
    }
    return model
