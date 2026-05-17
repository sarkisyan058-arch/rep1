[кейс7спринтаКартаДТП.py](https://github.com/user-attachments/files/27896227/7.py)

# **Что нужно сделать**
# 
# 
# Необходимо проверить, встречаются ли в данных дубликаты и пропуски. Это поможет заказчикам собирать более качественные данные.
# Также понадобится ответить на следующие вопросы:
# - как менялось число ДТП по временным промежуткам;
# - различается ли число ДТП для групп водителей с разным стажем.

# **Описание данных**
# 
# Описание данных
# **Датасеты Kirovskaya_oblast.csv, Moscowskaya_oblast.csv содержат информацию о ДТП:
# geometry.coordinates — координаты ДТП;**
# - id — идентификатор ДТП;
# - properties.tags — тег происшествия;
# - properties.light — освещённость;
# - properties.point.lat — широта;
# - properties.point.long — долгота;
# - properties.nearby — ближайшие объекты;
# - properties.region — регион;
# - properties.scheme — схема ДТП;
# - properties.address — ближайший адрес;
# - properties.weather — погода;
# - properties.category — категория ДТП;
# - properties.datetime — дата и время ДТП;
# - properties.injured_count — число пострадавших;
# - properties.parent_region — область;
# - properties.road_conditions — состояние покрытия;
# - properties.participants_count — число участников;
# - properties.participant_categories — категории участников.
# 
# 
# **Датасеты Moscowskaya_oblast_participiants.csv, Kirovskaya_oblast_participiants.csv хранят сведения об участниках ДТП:**
# - role — роль;
# - gender — пол;
# - violations — какие правила дорожного движения были нарушены конкретным участником;
# - health_status — состояние здоровья после ДТП;
# - years_of_driving_experience — число лет опыта;
# - id — идентификатор ДТП.
# 
# 
# **Датасеты Kirovskaya_oblast_vehicles.csv, Moscowskaya_oblast_vehicles.csv хранят сведения о транспортных средствах:**
# - year — год выпуска;
# - brand — марка транспортного средства;
# - color — цвет;
# - model — модель;
# - category — категория;
# - id — идентификатор ДТП.

# In[ ]:





# In[ ]:





# In[ ]:





# **Загружаем библиотеки**

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


# **Загружаем датасеты**

# In[10]:


# Загрузка данных
kirov_accidents = pd.read_csv('https://code.s3.yandex.net/datasets/Kirovskaya_oblast.csv')
moscow_accidents = pd.read_csv('https://code.s3.yandex.net/datasets/Moscowskaya_oblast.csv')
kirov_participants = pd.read_csv('https://code.s3.yandex.net/datasets/Kirovskaya_oblast_participiants.csv')
moscow_participants = pd.read_csv('https://code.s3.yandex.net/datasets/Moscowskaya_oblast_participiants.csv')
kirov_vehicles = pd.read_csv('https://code.s3.yandex.net/datasets/Kirovskaya_oblast_vehicles.csv')
moscow_vehicles = pd.read_csv('https://code.s3.yandex.net/datasets/Moscowskaya_oblast_vehicles.csv')

# Объединение данных по областям
accidents = pd.concat([kirov_accidents, moscow_accidents], ignore_index=True)
participants = pd.concat([kirov_participants, moscow_participants], ignore_index=True)
vehicles = pd.concat([kirov_vehicles, moscow_vehicles], ignore_index=True)

print("Размеры датасетов после объединения:")
display(pd.DataFrame({
    'Датасет': ['ДТП', 'Участники', 'Транспортные средства'],
    'Строк': [len(accidents), len(participants), len(vehicles)],
    'Столбцов': [len(accidents.columns), len(participants.columns), len(vehicles.columns)]
}))


# **Меняем названия столбцов**

# In[2]:


accidents.columns = [
    'geometry_coordinates', 'id', 'tags', 'light', 'lat', 'long',
    'nearby', 'region', 'scheme', 'address', 'weather', 'category',
    'datetime', 'injured_count', 'parent_region', 'road_conditions',
    'participants_count', 'participant_categories'
]

print("Первые 3 строки данных о ДТП после переименования столбцов:")
display(accidents.head(3))


# **Ищем пропуски**

# In[3]:


kirov_missing = kirov_accidents.isnull().sum()
kirov_missing_percent = (kirov_missing / len(kirov_accidents)) * 100

missing_df = pd.DataFrame({
    'Пропуски (шт.)': kirov_missing[kirov_missing > 0],
    'Процент пропусков': kirov_missing_percent[kirov_missing > 0].round(2)
})

print("Пропуски в данных Кировской области:")
display(missing_df)


# **Проверяем на дубликаты**

# In[4]:


# Явные дубликаты
accidents_duplicates = accidents.duplicated().sum()
print(f"Явные дубликаты: {accidents_duplicates} ({accidents_duplicates/len(accidents)*100:.2f}%)")

# Уникальность ID
id_unique = accidents['id'].nunique()
print(f"Уникальных ID: {id_unique} из {len(accidents)}")

duplicate_ids = accidents[accidents.duplicated(subset='id', keep=False)]
if not duplicate_ids.empty:
    print("\nПримеры дубликатов по ID:")
    display(duplicate_ids.head(5))
else:
    print("\nДубликатов по ID не обнаружено.")


# **Проверяем типы данных**

# In[5]:


print("Типы данных в датасете ДТП:")
display(accidents.dtypes.to_frame('Тип данных'))

# Преобразования типов
accidents['datetime'] = pd.to_datetime(accidents['datetime'])
accidents[['injured_count', 'participants_count']] = accidents[['injured_count', 'participants_count']].astype('Int64')
accidents[['light', 'weather', 'road_conditions']] = accidents[['light', 'weather', 'road_conditions']].astype('category')

print("\nТипы данных после преобразования:")
display(accidents.dtypes.to_frame('Тип данных'))


# ***Исследовательский анализ данных***

# **1. Распределение ДТП по дням недели и месяцам**

# In[6]:


accidents['day_of_week'] = accidents['datetime'].dt.day_name()
accidents['month'] = accidents['datetime'].dt.month_name()

# По дням недели
week_counts = accidents['day_of_week'].value_counts().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

plt.figure(figsize=(10, 6))
sns.barplot(x=week_counts.index, y=week_counts.values, palette='viridis')
plt.title('Число ДТП по дням недели')
plt.ylabel('Количество ДТП')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.show()

print("Статистика ДТП по дням недели:")
display(week_counts.to_frame('Количество ДТП'))

# По месяцам
month_counts = accidents['month'].value_counts().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])

plt.figure(figsize=(12, 6))
sns.barplot(x=month_counts.index, y=month_counts.values, palette='coolwarm')
plt.title('Число ДТП по месяцам')
plt.ylabel('Количество ДТП')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.show()

print("Статистика ДТП по месяцам:")
display(month_counts.to_frame('Количество ДТП'))


# **2. Анализ ДТП по стажу водителей**

# In[7]:


# Группировка стажа
def experience_group(years):
    if pd.isna(years):
        return 'Unknown'
    elif years < 2:
        return '0-2 года'
    elif years < 5:
        return '2-5 лет'
    elif years < 10:
        return '5-10 лет'
    else:
        return '10+ лет'

participants['experience_group'] = participants['years_of_driving_experience'].apply(experience_group)

# Подсчёт ДТП по группам стажа
experience_counts = participants.groupby('experience_group')['id'].count()

plt.figure(figsize=(8, 6))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
plt.pie(experience_counts.values, labels=experience_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Распределение ДТП по группам стажа водителей')
plt.show()

print("Распределение ДТП по стажу:")
display(experience_counts.to_frame('Количество ДТП'))

print("\nСтатистика стажа водителей:")
display(participants['years_of_driving_experience'].describe())


# In[21]:


print("Список столбцов в датасете ДТП:")
display(accidents.columns.tolist())


# **Анализ аварий по годам и регионам**

# In[30]:


# Исключение Москвы — создаём копии датафреймов
moscow_excluded = accidents[
    (accidents['properties.parent_region'] == 'Московская область') &
    (~accidents['properties.address'].str.contains('Москва', na=False))
].copy()
kirov_included = accidents[accidents['properties.parent_region'] == 'Кировская область'].copy()

print(f"Количество ДТП в Московской области (без Москвы): {len(moscow_excluded)}")
print(f"Количество ДТП в Кировской области: {len(kirov_included)}")

# Преобразование столбца с датой в формат datetime
moscow_excluded['properties.datetime'] = pd.to_datetime(
    moscow_excluded['properties.datetime'], errors='coerce'
)
kirov_included['properties.datetime'] = pd.to_datetime(
    kirov_included['properties.datetime'], errors='coerce'
)

# Извлечение года из даты для анализа динамики
moscow_excluded['year'] = moscow_excluded['properties.datetime'].dt.year
kirov_included['year'] = kirov_included['properties.datetime'].dt.year

# Подсчёт ДТП по годам для каждого региона
yearly_moscow = moscow_excluded.groupby('year').size()
yearly_kirov = kirov_included.groupby('year').size()

# Объединение данных для удобства сравнения
yearly_comparison = pd.DataFrame({
    'Московская область (без Москвы)': yearly_moscow,
    'Кировская область': yearly_kirov
}).fillna(0)

print("\nДинамика ДТП по годам:")
display(yearly_comparison)

# Визуализация динамики ДТП по годам
plt.figure(figsize=(12, 6))
plt.plot(yearly_comparison.index, yearly_comparison['Московская область (без Москвы)'],
         marker='o', label='Московская область (без Москвы)', linewidth=2)
plt.plot(yearly_comparison.index, yearly_comparison['Кировская область'],
         marker='s', label='Кировская область', linewidth=2)
plt.xlabel('Год')
plt.ylabel('Количество ДТП')
plt.title('Динамика числа ДТП по годам в Московской (без Москвы) и Кировской областях')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(yearly_comparison.index.astype(int))
plt.show()


# Расчёт относительных показателей на 100 000 жителей
population = {
    'Московская область': 8_000_000,
    'Кировская область': 1_200_000
}

total_moscow = len(moscow_excluded)
total_kirov = len(kirov_included)

rate_moscow = (total_moscow / population['Московская область']) * 100_000
rate_kirov = (total_kirov / population['Кировская область']) * 100_000

relative_analysis = pd.DataFrame({
    'Регион': ['Московская область (без Москвы)', 'Кировская область'],
    'Общее число ДТП': [total_moscow, total_kirov],
    'Население (чел.)': [population['Московская область'], population['Кировская область']],
    'ДТП на 100 000 жителей': [round(rate_moscow, 2), round(rate_kirov, 2)]
})

print("\nОтносительные показатели аварийности (на 100 000 жителей):")
display(relative_analysis)

# Визуализация относительных показателей
plt.figure(figsize=(8, 5))
bars = plt.bar(relative_analysis['Регион'], relative_analysis['ДТП на 100 000 жителей'],
             color=['#4e79a7', '#f28e2b'])
plt.title('Сравнение аварийности на 100 000 жителей')
plt.ylabel('Число ДТП на 100 000 жителей')
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()

# Краткий вывод
print("\nКраткий вывод:")
if rate_kirov > rate_moscow:
    print("Кировская область имеет более высокую аварийность в пересчёте на 100 000 жителей,")
    print("что может свидетельствовать о менее развитой инфраструктуре или других региональных факторах.")
else:
    print("Московская область (без Москвы) имеет более высокую аварийность на 100 000 жителей,")
    print("несмотря на потенциально лучшую инфраструктуру — возможно, из‑за высокой интенсивности движения.")


# In[ ]:





# In[ ]:





# In[ ]:





# 1. Качество данных
# Проблемы:
# 
# Пропуски. Наибольшее число пропусков — в столбцах nearby (ближайшие объекты) и scheme (схема ДТП). Также встречаются пропуски в данных о погоде, освещённости и координатах.
# 
# Дубликаты. Обнаружены явные дубликаты записей и дублирование по идентификаторам id.
# 
# Типы данных. Исходные типы данных не всегда оптимальны для анализа (например, даты хранились как строки).
# 
# Рекомендации:
# 
# внедрить валидацию данных на этапе ввода;
# 
# сделать обязательными ключевые поля (координаты, дата, ID);
# 
# настроить автоматическую проверку уникальности ID;
# 
# использовать геокодирование для восстановления координат по адресу.
# 
# 2. Динамика ДТП
# По дням недели:
# 
# пик аварийности — в пятницу (вероятно, из‑за усталости в конце рабочей недели и увеличения трафика);
# 
# минимум ДТП — в воскресенье (снижение общего трафика).
# 
# По месяцам:
# 
# максимум аварий — в летние месяцы (июнь–август): рост трафика, туристические поездки, ремонт дорог;
# 
# минимум — в зимние месяцы (январь–февраль): снижение активности из‑за сложных погодных условий.
# 
# 3. Распределение ДТП по стажу водителей
# Наиболее аварийные группы:
# 
# водители со стажем 0–2 года — из‑за недостатка опыта;
# 
# водители со стажем 10+ лет — возможно, из‑за излишней самоуверенности или снижения реакции с возрастом.
# 
# Наименьшая аварийность — у водителей со стажем 5–10 лет (оптимальный баланс опыта и осторожности).
# 
# 4. Сравнение регионов (Московская и Кировская области)
# Абсолютные показатели:
# 
# Московская область (без Москвы) — значительно больше ДТП из‑за высокой плотности населения и интенсивности движения.
# 
# Кировская область — меньше ДТП, но потенциально выше риски на отдельных участках.
# 
# Относительные показатели (на 100 000 жителей):
# 
# при учёте численности населения Кировская область может демонстрировать более высокий уровень аварийности из‑за:
# 
# худшего состояния дорог;
# 
# меньшей развитости инфраструктуры;
# 
# ограниченного доступа к экстренным службам.
# 
# Итоговые рекомендации
# Для улучшения качества данных:
# 
# автоматизировать сбор данных с датчиков и камер;
# 
# внедрить обязательные поля и валидацию в интерфейсе ввода;
# 
# регулярно проверять данные на дубликаты и пропуски.
# 
# Для снижения аварийности:
# 
# усилить контроль за новичками на дорогах (стаж 0–2 года);
# 
# проводить профилактические кампании для опытных водителей (10+ лет стажа);
# 
# уделить внимание безопасности в пиковые периоды (пятница, лето);
# 
# улучшить инфраструктуру в Кировской области (дороги, освещение, знаки).
# 
# Для дальнейшего анализа:
# 
# изучить влияние погодных условий и освещённости на аварийность;
# 
# проанализировать географию ДТП для выявления «очагов» аварийности;
# 
# сопоставить данные о нарушениях ПДД с категориями участников ДТП.
