from setuptools import setup, Extension

# Определяем расширение для модуля
arithmetic_brownian_motion_module = Extension(
    '_ArithmeticBrownianMotion',  # Имя с нижним подчеркиванием перед названием модуля
    sources=['ArithmeticBrownianMotion_wrap.cxx', 'ArithmeticBrownianMotion.cpp']  # Источники файлов
)

# Настроим setup.py для установки
setup(
    name='ArithmeticBrownianMotion',  # Имя модуля
    version='1.0',
    ext_modules=[arithmetic_brownian_motion_module],  # Модуль для расширения
)
