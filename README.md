# Быстрое обучение модели для распознавания документов на синтетическом датасете  

План проекта:  
  - Разработать пайплан генерации синтетического датасета по размеченному шаблону  
  - Сгенерировать датасет
  - Обучить модель, распознающую угол поворота паспорта на картинке
  - Обучить модель для нахождения текстовых полей
  - Обучить модель распознавания полей
  - * Сделать однопроходную модель, чтобы распознавать текст на документе сразу
  
Зачем синтетика?  
Два основных плюса:  
  1. Не надо размечать кучу картинок, доастаточно разметить шаблон и найти данные  
  2. Нет проблем с персональными данными!  
  
*DatasetGeneration/gen_passport_dataset.ipynb* - ipython notebook для генерации датасета  
*passport_rotator.ipynb* - ноутбук, в котором обучается модель для распознавания угла поворота от -30 до 30 градусов  
*passport_rotator_regressor.ipynb* - обучение регрессора для распознавания угла поворота, не работает (TODO)  
ссылка на датасеты: https://drive.google.com/drive/folders/1uL0QGjBCUDk77uvDDqQryLRg8Ckwn41b?usp=share_link
