from setuptools import setup, Extension

arithmetic_brownian_motion_module = Extension(
    '_ArithmeticBrownianMotionWrapper',
    sources=['ArithmeticBrownianMotionWrapper_wrap.cxx', 'ArithmeticBrownianMotionWrapper.cpp'] 
)

setup(
    name='ArithmeticBrownianMotionWrapper',
    version='1.0',
    ext_modules=[arithmetic_brownian_motion_module], 
)
