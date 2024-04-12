import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl


def reading_info(target, data_):
  ret_data = []
  for i in range(len(target)):
    for index, row in data_.iterrows():
      if str(target[i]).lower() in str(row[0]).lower().strip() :
        ret_data.append(row.to_list())
  return ret_data


def infaltion_type(year_inf):
  if year_inf < 6:
    return "низкая"
  elif 6 <= year_inf < 10:
    return "умеренная"
  elif 10 <= year_inf < 100:
    return "высокая"
  elif year_inf > 0:
    return "гиперинфляция"
  else:
    return "дефляция"


def cumulative_inflation_rates(start_year_salary, inflation):
  cum_inf_data = []
  local_inf = start_year_salary
  for i in range(len(inflation)):
    local_inf = local_inf * (1 + inflation[i]/100)
    cum_inf_data.append(local_inf)
  return cum_inf_data


def plot_cumulative_inflation_rates(nominal_data, cum_inf_industry, colors, lable):
  plt.plot(years, nominal_data, color=colors[0])
  plt.plot(years, cum_inf_industry, color=colors[1])
  plt.title(f'''Номинальная среднемесячная заработная плата \n vs заработная плата с накопленной инфляцией\n за период с 2000-2023 гг.\n в отрасли {lable}''')
  plt.xlabel('Год')
  plt.ylabel('Среднемесячная заработная плата (руб)')
  plt.legend(('Номинальная заработная плата', 'Заработная плата с накопленной инфляцией'), loc='best')
  plt.show()


def income_compare(nominal_data, cum_inf_industry, real_income, colors, lable):
  plt.plot(years, nominal_data, color=colors[0])
  plt.plot(years, cum_inf_industry, color=colors[1])
  plt.plot(years, real_income, color=colors[2])

  plt.title(f'''Номинальная среднемесячная заработная плата \n vs заработная плата с накопленной инфляцией\n за период с 2000-2023 гг.\n в отрасли {lable}''')
  plt.xlabel('Год')
  plt.ylabel('Среднемесячная заработная плата (руб)')
  plt.legend(('Номинальная заработная плата', 'Заработная плата с накопленной инфляцией', 'Реальная заработная плата'), loc='best')
  plt.show()



def infaltion_coef(inf):
  return 1 + inf / 100

url_slaries = 'https://github.com/dsdaria/InflationAnalyze/raw/main/salaries.xlsx'
df_2000_2016 = pd.read_excel(url_slaries, sheet_name="2000-2016 гг.")
df_2017 = pd.read_excel(url_slaries, sheet_name="с 2017 г.")

years = [int(i) for i in range(2000, 2023+1)]
work_names = ["Строительство", "Гостини", "Образование"]

date_2000 = ['type'] + [int(i) for i in range(2000, 2017)]
data_2000 = pd.DataFrame(reading_info(work_names, df_2000_2016), columns=date_2000)

date_2016 = ['type'] + [int(i) for i in range(2017, 2023+1)]
data_2016 = pd.DataFrame(reading_info(work_names, df_2017), columns=date_2016)

data_for_df = pd.merge(data_2000.iloc[:, 1:], data_2016.iloc[:, 1:], left_index=True, right_index=True)
labels = ["Строительство", "Гостиничный бизнес и общественное питание", "Образование"]
data_for_df.index = labels

url_inflation = 'https://github.com/dsdaria/InflationAnalyze/raw/main/inflation.xlsx'
inflation = pd.read_excel(url_inflation, sheet_name="инфляция")
year_inflation=[float(i) for i in inflation["Всего"][1:25].to_list()][::-1]

inflation_df = pd.DataFrame(data=year_inflation, columns=["Инфляция за год"])
inflation_df.index = years

inflation_df["Вид инфляции по тему роста"] = [infaltion_type(i) for i in inflation_df["Инфляция за год"]]

data = data_for_df.to_numpy()
year_inflation_coef = []
for i in year_inflation:
    year_inflation_coef.append(infaltion_coef(i))

df_inflation_coef = pd.DataFrame(data=year_inflation_coef, columns=["Коэффициент инфляции за год"])
df_inflation_coef.index = years

infaltion_coef_div = []
for i in year_inflation:
    infaltion_coef_div.append( 1 - i / 100)



real_income_building = np.multiply(data[0], infaltion_coef_div)
real_income_hotels = np.multiply(data[1], infaltion_coef_div)
real_income_education =  np.multiply(data[2], infaltion_coef_div)

df_inflation_coef['Реальный доход в строительстве'] = real_income_building
df_inflation_coef['Реальный доход в гостиничном бизнесе и общественном питании']= real_income_hotels
df_inflation_coef['Реальный доход в образовании'] = real_income_education

url_additional = 'https://github.com/dsdaria/InflationAnalyze/raw/main/additional.xlsx'
index = pd.read_excel(url_additional, index_col=None)

inf_data_df_ = df_inflation_coef[9:-1:].T
inf_data_df_chart = df_inflation_coef[9:-1:]

inf_data_df_["Показатель"] = inf_data_df_.index

glb_data = pd.concat([inf_data_df_, index][::-1], axis=0)
glb_data = glb_data.reset_index()
glb_data = glb_data.drop(['index'], axis=1)


def process_main_page():
    show_main_page(years, data_for_df)
    process_side_bar_inputs()

def show_main_page(years, data_for_df):
    st.title("Анализ заработной платы в России с 2000 года по 2023 год")

    st.markdown("Заработные платы в России с 2000 года по 2023 год в строительстве, гостиничном бизнесе и общественном питании и образование:")
    st.dataframe(data_for_df)
    st.markdown("Построим графики изменения номинальных заработных плат в упомянутых областях и определим, как они менялись в течение 23 лет.")
    plt.plot(years, data_for_df.iloc[0], color="orange")
    plt.plot(years, data_for_df.iloc[1], color="darkviolet")
    plt.plot(years, data_for_df.iloc[2], color="darkolivegreen")
    plt.title('''Изменение среднемесячной номинальной заработной платы\n за период с 2000 по 2023 гг.''')
    plt.xlabel('Год')
    plt.ylabel('Среднемесячная заработная плата (руб)')
    plt.legend(['Строительство', 'Гостиничный бизнес и общественное питание', 'Образование'], loc='best')
    st.pyplot(plt)
    st.markdown("**Вывод:** в общем случае наблюдается рост среднемесячной номинальной заработной платы за период с 2000 года по 2023 год за исключением некоторых малозначительных колебаний в отдельных отраслях.")


def infaltion(inflation_df, years, year_inflation):
    url = 'https://quote.rbc.ru/news/article/61e13fa79a79478207047ffc?from=copy'
    st.subheader("Инфляция")
    st.markdown("**Инфляция** — это темп устойчивого повышения общего уровня цен на товары и услуги за определенный промежуток времени, также инфляция показывает степень обесценивания денег. ")
    st.markdown("Чаще всего инфляцию принято указывать в годовом выражении, или, как еще говорят, год к году. ")
    st.markdown("Так, если инфляция в годовом выражении составила 8,4%,то имеют в виду, что набор одних и тех же товаров, который год назад стоил ₽100, сейчас стоит ₽108,4. Соответственно, ₽100 обесценились или потеряли покупательную способность на 8,4%. Это и есть инфляция. ")
    st.markdown("В России помимо годовой инфляции, Росстат измеряет еженедельную и ежемесячную.")
    st.markdown("Подробнее: %s " % url)
    st.markdown("Определим ***вид инфляции по темпам роста*** в каждый из рассматриваемых годов.")
    st.markdown("***Низкая*** - до 6 % в год.")
    st.markdown("***Умеренная*** - от 6 % до 10 % в год.")
    st.markdown("***Высокая (галопирующая)*** - от 10 % до 100 % в год.")
    st.markdown("***Гиперинфляция*** - цены растут на сотни и тысячи процентов, в особо тяжелых случаях люди отказываются от денег и переходят на бартер.")
    labels = ['высокая', 'умеренная', 'низкая']
    sizes = [inflation_df['Вид инфляции по тему роста'].value_counts()[labels[i]] for i in range(len(labels))]
    st.dataframe(inflation_df)

    fig1 = plt.figure(figsize=(9, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=["chocolate", "orange", "yellow"]);
    plt.title("Распределение видов инфляции\n по тему роста с 2000 года по 2023 год")
    st.pyplot(fig1)

    st.markdown("**Вывод:** практически в течение 12 лет (необязательно подряд идущих) в России наблюдалась высокая инфляция, низкая же встречалась лишь 5 раз из 24.")
    st.markdown("Рассмотрим, как менялась годовая инфляция за данный период времени.")

    fig2 = plt.figure(figsize=(9, 7))
    plt.plot(years, year_inflation, color="red")
    plt.title('''Изменение инфляции за период с 2000 по 2023 гг.''')
    plt.xlabel('Год')
    plt.ylabel('%')
    st.pyplot(fig2)

    fig3 = plt.figure(figsize=(9, 7))
    sns.histplot(data=inflation_df, x="Инфляция за год", bins=len(inflation_df), discrete=True, color='olive')
    st.pyplot(plt.gcf())

    st.markdown("**Вывод:** как видно из линейного графика и гистограммы, инфляция в России принимает в основном различные значения, начиная с 2000 года наблюдалось ее снижение, хотя и с колебаниями. В большинстве случаев (в 19 из 24) она не превышает 12%.")


def commulative_inflation(data, year_inflation, years):
    st.subheader("Накопленная инфляция")
    st.markdown("**Накопленная инфляция** - это изменение уровня цен на услуги и товары за определенный период времени (чаще годы).")
    st.markdown("Под 'инфляцией' обычно понимается рост цен относительно прошлого года. Однако в прошлогодний уровень включается инфляция предыдущего года и далее рекурсивно. Такой процесс в банковском деле называется сложным процентом, а в экономике — накопленной инфляцией.")
    st.markdown("***Формула накопленной инфляции:***")

    latext = r'''
    $$
        S = P * \prod_{k = 1}^{n}(1 + \frac{i_k}{100})
    $$
    '''
    st.write(latext)

    st.markdown("$S$ - цена или заработная плата через k лет с учетом накопленной инфляции")
    st.markdown("$P$ - первоначальная цена или заработная плата")
    st.markdown("$k$ - год в рассматриваемом промежутке: первый из временного отрезка обозначим через 1, а последний - через $n$")
    st.markdown("$i_{k}$ - инфляция в $k$-ый год")

    st.markdown("Для анализа в контексте данной задачи возьмем значения среднемесячных номинальных заработных плат для каждой анализируемой деятельности в 2000 году и найдем значения $S_{k}$ для каждого года и построим графики, сравнив с изменениями номинальных показателей для каждого года.")

    cum_inf_construction = cumulative_inflation_rates(data[0][0], year_inflation)
    cum_inf_hotels = cumulative_inflation_rates(data[1][0], year_inflation)
    cum_inf_education = cumulative_inflation_rates(data[2][0], year_inflation)

    fig_glb = plt.figure(figsize=(9, 7))
    plt.plot(years, cum_inf_construction, color='magenta')
    plt.plot(years, cum_inf_hotels, color='black')
    plt.plot(years, cum_inf_education, color='crimson')
    plt.title('Изменение среднемесячной номинальной заработной платы' + '\n' + 'c учетом накопленной инфляции' + '\n' + 'за период с 2000 по 2023 гг.')
    plt.xlabel('Год')
    plt.ylabel('Среднемесячная заработная плата руб')
    plt.legend (('Строительство', 'Гостиничный бизнес и общественное питание', 'Образование'), loc='best')
    st.pyplot(fig_glb)

    option = st.multiselect("Выберите отрасль", ("Строительство", "Гостиничный бизнес и общественное питание", "Образование"))

    if "Строительство" in option:
        fig_cons = plt.figure(figsize=(9, 7))
        plt.plot(years, data[0], color="orange")
        plt.plot(years, cum_inf_construction, color="magenta")
        plt.title(f'''Номинальная среднемесячная заработная плата \n vs \n заработная плата с накопленной инфляцией\n за период с 2000-2023 гг.\n в строительстве''')
        plt.xlabel('Год')
        plt.ylabel('Среднемесячная заработная плата (руб)')
        plt.legend(('Номинальная заработная плата', 'Заработная плата с накопленной инфляцией'), loc='best')
        st.pyplot(fig_cons)


    if "Гостиничный бизнес и общественное питание" in option:
        fig_hotels = plt.figure(figsize=(9, 7))
        plt.plot(years, data[1], color="darkviolet")
        plt.plot(years, cum_inf_hotels, color="black")
        plt.title(f'''Номинальная среднемесячная заработная плата \n vs \n заработная плата с накопленной инфляцией\n за период с 2000-2023 гг.\n в гостиничном бизнесе и общественном питании''')
        plt.xlabel('Год')
        plt.ylabel('Среднемесячная заработная плата (руб)')
        plt.legend(('Номинальная заработная плата', 'Заработная плата с накопленной инфляцией'), loc='best')
        st.pyplot(fig_hotels)

    if "Образование" in option:
        fig_edu = plt.figure(figsize=(9, 7))
        plt.plot(years, data[2], color="darkolivegreen")
        plt.plot(years, cum_inf_education, color="crimson")
        plt.title(f'''Номинальная среднемесячная заработная плата \n vs \n заработная плата с накопленной инфляцией\n за период с 2000-2023 гг.\n в образовании''')
        plt.xlabel('Год')
        plt.ylabel('Среднемесячная заработная плата (руб)')
        plt.legend(('Номинальная заработная плата', 'Заработная плата с накопленной инфляцией'), loc='best')
        st.pyplot(fig_edu)

    if len(option) != 0:
      st.markdown("**Вывод:** доход, полученный через 23 года по расчетам накопленной инфляции, растет, как и средний номинальный заработок.")



def real_income(year_inflation, years, data, df_inflation_coef):
    st.subheader("Расчет реальной заработной платы")
    st.markdown("***Реальный доход*** — доход, отражающий реальную покупательную способность денег, полученный с учетом инфляции.")
    st.markdown("Для того, чтобы найти реальный доход за каждый год, необходимо номинальный доход  этого года разделить на индекс инфляции:")

    latext_1 = r'''
    $$
         P = \dfrac{S}{I}
    $$
    '''
    st.write(latext_1)
    st.markdown("***Индекс инфляции:***")


    latext_2 = r'''
    $$
         I = 1 + \frac{i}{100}
    $$
    '''
    st.write(latext_2)

    st.markdown("Также можно провести расчеты с помощью другой формулы:")

    latext_3 = r'''
    $$
         P = S * (1 - \frac{i}{100})
    $$
    '''
    st.write(latext_3)

    st.markdown("В обоих случаях:")
    st.markdown("$P$ - реальный доход")
    st.markdown("$S$ - номинальный доход")
    st.markdown("$i$ - инфляция в рассматриваемый год")

    st.dataframe(df_inflation_coef)

    fig_real= plt.figure(figsize=(9, 7))
    plt.plot(years, df_inflation_coef['Реальный доход в строительстве'], color="tomato")
    plt.plot(years, df_inflation_coef['Реальный доход в гостиничном бизнесе и общественном питании'], color="gold")
    plt.plot(years, df_inflation_coef['Реальный доход в образовании'], color="chartreuse")
    plt.title('''Изменение реальной заработной платы\n за период с 2000 по 2023 гг.''')
    plt.xlabel('Год')
    plt.ylabel('Среднемесячная заработная плата (руб)')
    plt.legend(['Строительство', 'Гостиничный бизнес и общественное питание', 'Образование'], loc='best')
    st.pyplot(fig_real)

    cum_inf_construction = cumulative_inflation_rates(data[0][0], year_inflation)
    cum_inf_hotels = cumulative_inflation_rates(data[1][0], year_inflation)
    cum_inf_education = cumulative_inflation_rates(data[2][0], year_inflation)

    option_r = st.multiselect("Выберите отрасль для сравнения", ("Строительство", "Гостиничный бизнес и общественное питание", "Образование"))

    if "Строительство" in option_r:
        fig_cons_r = plt.figure(figsize=(9, 7))
        plt.plot(years, data[0], color="orange")
        plt.plot(years, cum_inf_construction, color="magenta")
        plt.plot(years, real_income_building, color="tomato")
        plt.title("Сравнение заработной платы в строительстве за период 2000-2023 гг.")
        plt.xlabel('Год')
        plt.ylabel('Среднемесячная заработная плата (руб)')
        plt.legend(('Номинальная заработная плата', 'Заработная плата с накопленной инфляцией', 'Реальная заработная плата'), loc='best')

        st.pyplot(fig_cons_r)


    if "Гостиничный бизнес и общественное питание" in option_r:
        fig_hotels_r = plt.figure(figsize=(9, 7))
        plt.plot(years, data[1], color="darkviolet")
        plt.plot(years, cum_inf_hotels, color="black")
        plt.plot(years, real_income_hotels, color="gold")
        plt.title("Сравнение заработной платы в гостиничном бизнесе и общественном питании за период 2000-2023 гг.")
        plt.xlabel('Год')
        plt.ylabel('Среднемесячная заработная плата (руб)')
        plt.legend(('Номинальная заработная плата', 'Заработная плата с накопленной инфляцией', 'Реальная заработная плата'), loc='best')

        st.pyplot(fig_hotels_r)


    if "Образование" in option_r:
        fig_edu_r = plt.figure(figsize=(9, 7))
        plt.plot(years, data[2], color="darkolivegreen")
        plt.plot(years, cum_inf_education, color="crimson")
        plt.plot(years, real_income_education, color="chartreuse")
        plt.title("Сравнение заработной платы в образовании за период 2000-2023 гг.")
        plt.xlabel('Год')
        plt.ylabel('Среднемесячная заработная плата (руб)')
        plt.legend(('Номинальная заработная плата', 'Заработная плата с накопленной инфляцией', 'Реальная заработная плата'), loc='best')

        st.pyplot(fig_edu_r)

    if len(option_r) != 0:
      st.markdown("**Вывод:** как видно из вышепредставленных графиков, несмотря на наличие инфляции более 10% в течение 17 лет, реальные доходы в рассматриваемых отраслях растут.")


def demography(glb_data, inflation_df):
    st.subheader("Связь заработной платы и демографических показателей")
    st.markdown("В дополнение рассмотрим связь (определим, существует ли она вообще) ВВП на душу населения и некоторых демографических показателей на уровень реальных заработных плат*. Представим изменения с помощью графиков. Все данные взяты с официального сайта Росстата. По причине наличия однородных данных с 2009 года, рассмотрим временной отрезок с 2009 года по 2022 год. Данные вручную были добавлены в таблицу excel из разных таблиц, ссылки на которые приведены ниже.")

    st.markdown("***Для анализа были выбраны следующие показатели:***")
    st.markdown("1. ВВП на душу населения")
    st.markdown("2. Численность населения")
    st.markdown("3. Численность городского населения")
    st.markdown("4. Численность сельского населения")
    st.markdown("5. Уровень безработицы, %")
    st.markdown("6. Уровень безработицы среди мужчин, %")
    st.markdown("7. Уровень безработицы среди женщин, %")

    st.markdown("***Валово́й вну́тренний проду́кт (англ. gross domestic product), общепринятое сокращение — ВВП (англ. GDP)*** — макроэкономический показатель, отражающий рыночную стоимость всех конечных товаров и услуг (то есть предназначенных для непосредственного употребления, использования или применения), произведённых за год во всех отраслях экономики на территории конкретного государства для потребления, экспорта и накопления, вне зависимости от национальной принадлежности использованных факторов производства.")
    url_1 = 'https://ru.wikipedia.org/wiki/Валовой_внутренний_продукт'
    st.markdown("Источник: %s " % url_1)

    years_comp = [int(i) for i in range(2009, 2022+1)]
    glb_data.index = glb_data["Показатель"]
    glb_data = glb_data.T
    glb_data = glb_data.drop(['Показатель'])
    st.markdown("*Примечание:* здесь и далее под 'ВВП' подразумевается ВВП на душу населения.")
    st.dataframe(glb_data)

    st.markdown("Изменение численности населения и уровень безработицы в России с 2009 года по 2022 год")
    option_demo = st.multiselect("Выберите показатель для анализа", ("Численность населения", "Уровень безработицы"))

    if "Численность населения" in option_demo:
        fig_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 8));
        fig_.tight_layout(pad=6);
        fig_.suptitle('Изменение численности населения России с 2009 года по 2022 год')
        ax1.plot(glb_data["Население"], color="navy")
        ax1.set_title("Население");
        ax1.set_title("Население");
        ax2.plot(glb_data["Городское население"], color="navy")
        ax2.set_title("Городское население");
        ax3.plot(glb_data["Сельское население"], color="navy")
        ax3.set_title("Сельское население");
        st.pyplot(fig_)

    if "Уровень безработицы" in option_demo:
        figur, (ax1_0, ax2_0, ax3_0) = plt.subplots(1, 3, figsize=(10, 10));
        figur.tight_layout(pad=8);
        figur.suptitle('Уровень безработицы в России с 2009 года по 2022 год');
        ax1_0.plot(glb_data["Уровень безработицы, % "], color="navy");
        ax1_0.set_title("Уровень безработицы,\n %");
        ax1_0.set_xlabel('Год');
        ax1_0.set_ylabel('%');
        ax2_0.plot(glb_data["Уровень безработицы среди мужчин, % "], color="navy")
        ax2_0.set_title("Уровень безработицы\n среди мужчин, % ");
        ax2_0.set_xlabel('Год');
        ax2_0.set_ylabel('%');
        ax3_0.plot(glb_data["Уровень безработицы среди женщин, % "], color="navy");
        ax3_0.set_title("Уровень безработицы\n среди женщин, % ");
        ax3_0.set_xlabel('Год');
        ax3_0.set_ylabel('%');
        st.pyplot(figur)

    st.markdown("Возможно построить график для сравнения изменения показателей между 2009 и 2022 годами включительно, однако не у всех из них единицы измерения совпадает.")
    st.markdown("Будьте внимательны при проведении сравнения!")
    st.markdown("Ниже указаны предлагаемые показатели и их единицы измерения.")
    st.markdown("**Сравниваемые показатели:**")
    criteria_ = ['Реальный доход в строительстве (руб.)', 'Реальный доход в гостиничном бизнесе и общественном питании (руб.)',  'Реальный доход в образовании (руб.)',
    'Инфляция (%)', 'ВВП (руб.)', 'Население (млн. чел.)', 'Городское население (млн. чел.)', 'Уровень безработицы среди женщин (%)', 'Уровень безработицы среди мужчин (%)']
    possible_criteria = pd.DataFrame(criteria_, columns=['Показатель и единица измерения'])
    possible_criteria.index = np.arange(1, len(possible_criteria)+1)
    st.dataframe(possible_criteria)


    criteria1 = st.multiselect(
    'Выберите отрасль для сравнения доходов',
    ['Реальный доход в строительстве', 'Реальный доход в гостиничном бизнесе и общественном питании',  'Реальный доход в образовании'])

    criteria2 = st.multiselect(
    'Выберите показатели для сравнения',
    ['Инфляция', 'Уровень безработицы среди женщин, % ', 'Уровень безработицы среди мужчин, % '])

    criteria = criteria1 + criteria2

    criteria_colors = ["red", "indigo", "green", "navy", "orange", "magenta", "lime"]
    criteria_colors_ = ["красный", "индиго", "зеленый", "синий", "оранжевый", "пурпурный", "лайм"]

    criteria_for_df = []

    fig_comp_1, ax_comp_1 = plt.subplots();
    ax_comp_2 = ax_comp_1.twinx();
    fig_comp_1.suptitle("Совместное изменение заработных плат с инфляцией\n и уровнем безработицы с 2009 года по 2022 год.");

    lcl_data = (inflation_df["Инфляция за год"][9:-1])
    st.markdown("***Сравниваемые показатели - цвет графика:***")
    ax_comp_1.set_ylabel('Руб.')
    ax_comp_1.set_xlabel('Год')
    for i in range(len(criteria)):
        if criteria[i] in  ['Реальный доход в строительстве', 'Реальный доход в гостиничном бизнесе и общественном питании',  'Реальный доход в образовании']:
            ax_comp_1.plot(years_comp, glb_data[criteria[i]], color=criteria_colors[i])
            names = criteria[i] + " - " + criteria_colors_[i]
            st.write(names)
        else:
          if criteria[i] == 'Инфляция':
              ax_comp_2.plot(years_comp, lcl_data, color=criteria_colors[i], marker='o')
              ax_comp_2.set_ylabel('%')
              name_inf = "Инфляция" + " - " + criteria_colors_[i]
              st.write(name_inf)

          elif criteria[i] in ['Уровень безработицы, % ', 'Уровень безработицы среди женщин, % ', 'Уровень безработицы среди мужчин, % ']:
             ax_comp_2.plot(years_comp, glb_data[criteria[i]], color=criteria_colors[i])
             name_inf_ = criteria[i] + " - " + criteria_colors_[i]
             st.write(name_inf_)
             ax_comp_2.set_ylabel('%')


    st.pyplot(fig_comp_1)

    criteria1_pop = st.multiselect(
    'Выберите отрасль для сравнения доходов с численностью населения',
    ['Реальный доход в строительстве', 'Реальный доход в гостиничном бизнесе и общественном питании',
    'Реальный доход в образовании'])

    criteria2_pop = st.multiselect(
    'Выберите категорию населения для сравнения',
    ['Население', 'Городское население', 'Сельское население'])

    criteria_pop = criteria1_pop + criteria2_pop

    criteria_colors_pop_ = ["red", "indigo", "green", "navy", "orange", "black"]
    criteria_colors_pop = ["красный", "индиго", "зеленый", "синий", "оранжевый", "черный"]

    criteria_for_df_pop = []

    fig_comp_1_pop, ax_comp_1_pop = plt.subplots();
    ax_comp_2_pop = ax_comp_1_pop.twinx();
    fig_comp_1_pop.suptitle("Совместное изменение заработных плат с численностью населения 2009 года по 2022 год.");

    lcl_data_pop = (inflation_df["Инфляция за год"][9:-1])
    st.markdown("***Сравниваемые показатели - цвет графика:***")
    ax_comp_1_pop.set_ylabel('Руб.')
    ax_comp_1_pop.set_xlabel('Год')
    for i in range(len(criteria_pop)):
        if criteria_pop[i] in  ['Реальный доход в строительстве', 'Реальный доход в гостиничном бизнесе и общественном питании',  'Реальный доход в образовании']:
            ax_comp_1_pop.plot(years_comp, glb_data[criteria_pop[i]], color=criteria_colors_pop_[i])
            names = criteria_pop[i] + " - " + criteria_colors_pop[i]
            st.write(names)
        else:
          if criteria_pop[i] in ['Население', 'Городское население', 'Сельское население']:
              ax_comp_2_pop.plot(years_comp, glb_data[criteria_pop[i]], color=criteria_colors_pop_[i])
              ax_comp_2_pop.set_ylabel('Млн. чел.')
              names = criteria_pop[i] + " - " + criteria_colors_pop[i]
              st.write(names)


    st.pyplot(fig_comp_1_pop)

    criteria_GDP = st.multiselect(
    'Выберите отрасль для сравнения доходов с ВВП',
    ['Реальный доход в строительстве', 'Реальный доход в гостиничном бизнесе и общественном питании',
    'Реальный доход в образовании'])

    criteria_colors_GDP_ = ["indigo", "green", "navy", "orange", "black", "lime", "fuchsia"]
    criteria_colors_GDP = ["индиго", "зеленый", "синий", "оранжевый", "черный", "лайм", "фуксия"]

    criteria_for_df_GDP = criteria_GDP

    fig_comp_1_GDP, ax_comp_1_GDP = plt.subplots();
    ax_comp_2_GDP = ax_comp_1_GDP.twinx();
    fig_comp_1_GDP.suptitle("Совместное изменение заработных плат с ВВП 2009 года по 2022 год.");

    st.markdown("***Сравниваемые показатели - цвет графика:***")
    name_GDP = "ВВП"+ " - " + "красный"
    ax_comp_2_GDP.plot(years_comp, glb_data["ВВП"], color="red", marker="s")
    ax_comp_2_GDP.set_ylabel('Млн. руб.')
    ax_comp_1_GDP.set_ylabel('Руб.')
    ax_comp_1_GDP.set_xlabel('Год')

    st.write(name_GDP)
    for i in range(len(criteria_GDP)):
        if criteria_GDP[i] in  ['Реальный доход в строительстве', 'Реальный доход в гостиничном бизнесе и общественном питании',  'Реальный доход в образовании']:
            ax_comp_1_GDP.plot(years_comp, glb_data[criteria_GDP[i]], color=criteria_colors_GDP_[i])

            names = criteria_GDP[i] + " - " + criteria_colors_GDP[i]
            st.write(names)

    st.pyplot(fig_comp_1_GDP)
    st.markdown("**Вывод:** как видно из графиков выше, при повышении реального дохода во всех трех рассматриваемых отраслях (образование, строительство, гостиничный бизнес и общественное питание), уровень безработицы снижается. Это можно объяснить тем, что в образовании и сфере обслуживания в основном задействованы женщины. Их привлекают растущие заработные платы, в связи с чем снижается процент безработных. В строительстве же работают преимущественно мужчины, которых также привлекают растущие зарплаты (процент безработных в данной демографической группе также снижается). Рост заработной платы в строительстве также можно объяснить спросом на жилье, так как число проживающих в сельской местности снижается, соответственно эти люди перебираются в города (городское население увеличивается). Соответственно нужны те, кто построит новые объекты жилья. Значения ВВП на душу населения и реальных заработных плат в гостиничном бизнесе и общественном питании растут вместе, так как данная деятельность относят к категории 'услуги', которые в свою очередь увеличивают значение ВВП на душу населения.")
    st.markdown("**Четких связей выявлено не было.**")

    st.markdown("****Источники данных для выявления связи заработной платы и демографических показателей:***")
    url_gdp = 'https://rosstat.gov.ru/statistics/accounts'
    st.markdown("Таблица 'ВВП на душу населения': %s" % url_gdp)
    url_pop = 'https://rosstat.gov.ru/folder/12781'
    st.markdown("Таблица 'Численность населения': %s" % url_pop)
    url_index = 'https://rosstat.gov.ru/labor_market_employment_salaries'
    st.markdown("Таблица 'Индикаторы достойного труда': %s" % url_index)

def process_side_bar_inputs():
    st.sidebar.header("Тип анализа:")
    infaltion_ = st.sidebar.checkbox('Инфляция')
    commulative_inflation_ = st.sidebar.checkbox('Накопленная инфляция')
    real_income_ = st.sidebar.checkbox('Расчет реальной заработной платы')
    demography_ = st.sidebar.checkbox('Связь заработной платы и демографических показателей')


    if infaltion_:
        infaltion(inflation_df, years, year_inflation)

    if commulative_inflation_:
        commulative_inflation(data, year_inflation, years)

    if real_income_:
        real_income(year_inflation, years, data, df_inflation_coef)


    if demography_:
        demography(glb_data, inflation_df)

if __name__ == "__main__":
    process_main_page()
