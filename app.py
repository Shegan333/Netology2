"""Итоговая работа: streamlit run app.py (Python 3.11+).

Единица исходной таблицы orders — товарная позиция заказа. Число заказов
считается через nunique(order_id), денежные суммы — в целых копейках.
"""

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")  # Работа без графического окружения на сервере.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
FILES = ("orders.csv", "users.csv", "items.csv")
REQUIRED = {
    "orders": ["order_id", "user_id", "order_date", "item_id", "quantity", "price_per_unit"],
    "users": ["user_id", "user_name", "registration_date", "city", "user_segment"],
    "items": ["item_id", "item_name", "category", "supplier", "base_price"],
}
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
DAY_NAMES = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
BLUE, INK, MUTED, GRID = "#235C9B", "#172B42", "#526378", "#E3E9EF"


def number(value, decimals=0):
    """Русское форматирование без зависимости от системной locale."""
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def rubles(value):
    return f"{number(value, 2)} ₽"


@st.cache_data(show_spinner="Загрузка CSV…")
def load_sources(directory, signature):
    """signature=(имя, mtime, размер) обновляет кеш при изменении файлов."""
    return {
        Path(filename).stem: pd.read_csv(Path(directory) / filename, encoding="utf-8-sig")
        for filename in FILES
    }


def to_number(series):
    cleaned = series.astype("string").str.strip().str.replace("\u00a0", "", regex=False)
    cleaned = cleaned.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce").replace([np.inf, -np.inf], np.nan)


def to_id(series):
    values = to_number(series)
    return values.where((values > 0) & (values.mod(1) == 0)).astype("Int64")


def prepare_data(sources):
    """Очистка копий данных и LEFT JOIN с контролем кардинальности m:1.

    Неизвестные пользователи/товары сохраняют продажи. Неверные даты,
    количества и цены исключаются с доступным отчётом о причинах.
    Конфликтующие ключи справочника вызывают явную ошибку вместо удвоения сумм.
    """
    tables, duplicates = {}, {}
    for name, columns in REQUIRED.items():
        frame = sources[name].copy()
        frame.columns = frame.columns.str.strip()
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}.csv: отсутствуют столбцы {', '.join(sorted(missing))}.")
        # В расчёт не попадают неописанные дополнительные поля CSV.
        frame = frame[columns]
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            frame[column] = frame[column].astype("string").str.strip().replace("", pd.NA)
        before = len(frame)
        frame = frame.drop_duplicates().copy()
        duplicates[name] = before - len(frame)
        tables[name] = frame

    orders, users, items = (tables[key] for key in ("orders", "users", "items"))
    for frame, keys in ((orders, ["order_id", "user_id", "item_id"]),
                        (users, ["user_id"]), (items, ["item_id"])):
        for key in keys:
            frame[key] = to_id(frame[key])

    bad_keys = {}
    for name, frame, key in (("users", users, "user_id"), ("items", items, "item_id")):
        bad_keys[name] = int(frame[key].isna().sum())
        frame.dropna(subset=[key], inplace=True)
        # После нормализации '1' и '1.0' становятся одним ключом.
        previous_size = len(frame)
        frame.drop_duplicates(inplace=True)
        duplicates[name] += previous_size - len(frame)
        if frame[key].duplicated().any():
            raise ValueError(
                f"{name}.csv: одному {key} соответствуют разные записи. "
                "Исправьте конфликтующие строки справочника."
            )

    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce", format="mixed")
    users["registration_date"] = pd.to_datetime(users["registration_date"], errors="coerce", format="mixed")
    orders["quantity"] = to_number(orders["quantity"])
    orders["price_per_unit"] = to_number(orders["price_per_unit"])
    items["base_price"] = to_number(items["base_price"])
    items["base_price"] = items["base_price"].where(items["base_price"] >= 0)
    reasons = pd.DataFrame({
        "Некорректный order_id": orders["order_id"].isna(),
        "Некорректная дата": orders["order_date"].isna(),
        "Количество должно быть целым и > 0": (
            orders["quantity"].isna() | (orders["quantity"] <= 0) | (orders["quantity"].mod(1) != 0)
        ),
        "Цена отсутствует или < 0": orders["price_per_unit"].isna() | (orders["price_per_unit"] < 0),
    })
    invalid = reasons.any(axis=1)
    rejected = orders.loc[invalid].copy()
    rejected["Причина"] = reasons.loc[invalid].apply(
        lambda row: "; ".join(row.index[row]), axis=1
    ) if invalid.any() else pd.Series(dtype="string")
    orders = orders.loc[~invalid].copy()

    # У всех позиций одного заказа должны совпадать покупатель и время заказа.
    identity = orders.groupby("order_id")[["user_id", "order_date"]].nunique(dropna=False)
    if (identity > 1).any().any():
        raise ValueError("orders.csv: у позиций одного order_id различаются покупатель или дата заказа.")
    orders["quantity"] = orders["quantity"].astype("int64")
    # Сначала округляем цену единицы до копеек, затем умножаем на количество.
    orders["price_kopecks"] = orders["price_per_unit"].mul(100).round().astype("int64")
    orders["price_per_unit"] = orders["price_kopecks"] / 100
    orders["revenue_kopecks"] = orders["quantity"] * orders["price_kopecks"]

    merged = orders.merge(users, how="left", on="user_id", validate="many_to_one", indicator="user_join")
    merged = merged.merge(items, how="left", on="item_id", validate="many_to_one", indicator="item_join")
    unmatched_users = int(merged["user_join"].eq("left_only").sum())
    unmatched_items = int(merged["item_join"].eq("left_only").sum())
    missing_before_fill = merged.isnull().sum().rename("Пропуски до заполнения")
    for column in ["user_name", "city", "user_segment", "category", "supplier"]:
        merged[column] = merged[column].fillna("Неизвестно")
    fallback_name = "Товар #" + merged["item_id"].astype("string").fillna("без ID")
    merged["item_name"] = merged["item_name"].fillna(fallback_name)
    merged["revenue"] = merged["revenue_kopecks"] / 100
    merged["weekday"] = merged["order_date"].dt.dayofweek
    merged.sort_values(["order_date", "order_id"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    report = {
        "duplicates": duplicates, "bad_keys": bad_keys, "rejected": rejected,
        "unmatched_users": unmatched_users, "unmatched_items": unmatched_items,
        "missing_before_fill": missing_before_fill,
        "invalid_registration_dates": int(users["registration_date"].isna().sum()),
        "missing_base_prices": int(items["base_price"].isna().sum()),
    }
    return merged, report


def filter_data(frame, start, end, categories):
    """Конец периода включён целиком, в том числе время 23:59:59."""
    dates = frame["order_date"].dt.normalize()
    return frame.loc[
        dates.between(pd.Timestamp(start), pd.Timestamp(end)) & frame["category"].isin(categories)
    ].copy()


def metrics(frame):
    order_count = int(frame["order_id"].nunique())
    revenue = int(frame["revenue_kopecks"].sum()) / 100
    return {
        "orders": order_count, "revenue": revenue,
        "users": int(frame["user_id"].nunique()),
        "average_order": revenue / order_count if order_count else 0.0,
    }


def aggregate_data(frame, start, end):
    """groupby для товаров/категорий, pivot_table для дней недели."""
    products = frame.groupby(["item_id", "item_name"], dropna=False).agg(
        revenue_kopecks=("revenue_kopecks", "sum"), quantity=("quantity", "sum"),
        orders=("order_id", "nunique"),
    ).reset_index().sort_values(["revenue_kopecks", "item_name"], ascending=[False, True])
    categories = frame.groupby("category")["revenue_kopecks"].sum().sort_values(ascending=False)
    calendar = pd.date_range(start, end, freq="D")
    exposures = pd.Series(calendar.dayofweek).value_counts().reindex(range(7), fill_value=0)
    if frame.empty:
        counts = pd.Series(0, index=range(7), dtype="int64")
    else:
        pivot = frame.pivot_table(index="weekday", values="order_id", aggfunc="nunique")
        counts = pivot["order_id"].reindex(range(7), fill_value=0).astype(int)
    weekdays = pd.DataFrame({"orders": counts, "days": exposures})
    weekdays["average"] = weekdays["orders"].div(weekdays["days"].replace(0, np.nan))
    daily = frame.groupby(frame["order_date"].dt.normalize()).agg(
        revenue_kopecks=("revenue_kopecks", "sum"), orders=("order_id", "nunique")
    ).reindex(calendar, fill_value=0)
    return products, categories, weekdays, daily


def conclusions(frame, start, end):
    """Текст пересчитывается для текущих фильтров; ничьи не скрываются."""
    if frame.empty:
        return []
    products, categories, weekdays, _ = aggregate_data(frame, start, end)
    total = int(frame["revenue_kopecks"].sum())
    findings = []
    if total > 0:
        leaders = categories[categories.eq(categories.max())]
        names = ", ".join(leaders.index)
        findings.append(
            f"**Категории с наибольшей выручкой: {names}.** "
            f"{rubles(leaders.iloc[0] / 100)} и {number(leaders.iloc[0] / total * 100, 1)}% "
            "выручки выбранной выборки на каждую категорию. "
            "При планировании ассортимента проверьте наличие этих товаров; прибыльность требует данных о затратах."
        )
        leaders = products.loc[products["revenue_kopecks"].eq(products["revenue_kopecks"].max())]
        findings.append(
            f"**Лидеры товаров по выручке: {', '.join(leaders['item_name'])}.** "
            f"На каждый приходится {rubles(leaders.iloc[0]['revenue_kopecks'] / 100)} "
            f"({number(leaders.iloc[0]['revenue_kopecks'] / total * 100, 1)}% выручки выборки)."
        )
    else:
        findings.append("**Выручка выборки равна нулю.** Доли категорий и лидерство по выручке не определяются.")
    unit_leaders = products.loc[products["quantity"].eq(products["quantity"].max())]
    findings.append(
        f"**Лидеры по проданным единицам: {', '.join(unit_leaders['item_name'])}.** "
        f"У каждого — {number(unit_leaders.iloc[0]['quantity'])} шт."
    )
    available = weekdays["average"].dropna()
    best = available[available.eq(available.max())]
    findings.append(
        f"**Наибольшее среднее число заказов за день: {', '.join(DAY_NAMES[i] for i in best.index)}.** "
        f"{number(best.iloc[0], 2)} заказа на календарный день соответствующего типа. "
        "Короткий период не позволяет утверждать, что это устойчивая недельная закономерность."
    )
    customer_orders = frame.dropna(subset=["user_id"]).groupby("user_id")["order_id"].nunique()
    if len(customer_orders):
        repeat = int(customer_orders.gt(1).sum())
        findings.append(
            f"**{repeat} из {len(customer_orders)} покупателей сделали больше одного заказа "
            f"в выборке ({number(repeat / len(customer_orders) * 100, 1)}%).** "
            "Это доля повторных покупателей внутри периода и категорий, а не показатель удержания."
        )
    return findings


def figure(size=(10, 4)):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
        "axes.edgecolor": GRID, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    ax.set_axisbelow(True)
    return fig, ax


def show_figure(fig):
    st.pyplot(fig, clear_figure=True, width="stretch")
    plt.close(fig)


def barh_chart(labels, values, suffix=" ₽", shares=None):
    fig, ax = figure((11, max(3.5, len(labels) * 0.55)))
    labels = [textwrap.fill(str(label), 31) for label in labels]
    bars = ax.barh(range(len(labels)), values, color=BLUE, height=0.62)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    maximum = max(values, default=0)
    ax.set_xlim(0, maximum * 1.40 if maximum else 1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: number(x)))
    ax.set_xlabel("Выручка, ₽")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    annotations = [
        number(value, 2) + suffix + (f" · {number(shares[i], 1)}%" if shares is not None else "")
        for i, value in enumerate(values)
    ]
    ax.bar_label(bars, labels=annotations, padding=6, fontsize=10, color=INK)
    show_figure(fig)


def show_quality(sources, frame, report):
    st.subheader("Очистка и объединение")
    summary = pd.DataFrame({
        "Источник": list(REQUIRED),
        "Строк в CSV": [len(sources[name]) for name in REQUIRED],
        "Удалено точных дублей": [report["duplicates"][name] for name in REQUIRED],
    })
    st.dataframe(summary, hide_index=True, width="stretch")
    st.write(
        f"После очистки: **{len(frame)} позиций**. Исключено некорректных позиций: "
        f"**{len(report['rejected'])}**. Не найдено в справочнике пользователей: "
        f"**{report['unmatched_users']} позиций**, товаров: **{report['unmatched_items']} позиций**."
    )
    st.caption("Отчёт о качестве относится ко всем исходным файлам, независимо от фильтров.")
    st.markdown(
        "Объединение: `orders LEFT JOIN users` по `user_id`, затем `LEFT JOIN items` по `item_id`. "
        "Проверка `many_to_one` защищает от размножения продаж. Неизвестные атрибуты заполнены "
        "как «Неизвестно», отсутствующее название — ID товара. Пропущенные даты регистрации "
        "и справочные цены остаются пропусками и не влияют на выручку. Точные дубли удалены."
    )
    st.write(
        f"Некорректные ключи справочников, исключённые из объединения: "
        f"users — {report['bad_keys']['users']}, items — {report['bad_keys']['items']}. "
        f"Пропуски/ошибки дат регистрации: {report['invalid_registration_dates']}; "
        f"справочной цены: {report['missing_base_prices']}."
    )
    nulls = report["missing_before_fill"].to_frame()
    nulls["Пропуски после подготовки"] = frame.isnull().sum()
    nulls["Тип после подготовки"] = frame.dtypes.astype(str)
    st.dataframe(nulls, width="stretch")
    if not report["rejected"].empty:
        st.dataframe(report["rejected"], hide_index=True, width="stretch")


def main():
    st.set_page_config(page_title="Анализ интернет-магазина", layout="wide")
    st.title("Анализ интернет-магазина")
    try:
        signature = tuple((name, (ROOT / name).stat().st_mtime_ns, (ROOT / name).stat().st_size) for name in FILES)
        sources = load_sources(str(ROOT), signature)
        frame, report = prepare_data(sources)
    except FileNotFoundError as error:
        st.error(f"Не найден файл {Path(error.filename).name}. Положите {', '.join(FILES)} рядом с app.py.")
        st.stop()
    except (ValueError, pd.errors.ParserError, UnicodeError, OverflowError) as error:
        st.error(f"Не удалось подготовить данные: {error}")
        st.info("Проверьте заголовки, значения и кодировку UTF-8 в исходных CSV, затем обновите страницу.")
        st.stop()
    if frame.empty:
        st.warning("После очистки не осталось корректных заказов. Проверьте исходные файлы и причины исключения.")
        show_quality(sources, frame, report)
        st.stop()

    minimum, maximum = frame["order_date"].min().date(), frame["order_date"].max().date()
    all_categories = sorted(frame["category"].unique())
    def reset_filters():
        st.session_state["period"] = (minimum, maximum)
        st.session_state["categories"] = all_categories

    if st.session_state.get("data_signature") != signature:
        reset_filters()
        st.session_state["data_signature"] = signature
    with st.sidebar:
        st.header("Фильтры")
        selected_dates = st.date_input("Период заказов", min_value=minimum,
                                       max_value=maximum, format="DD.MM.YYYY", key="period")
        categories = st.multiselect("Категории", all_categories, key="categories")
        st.button("Сбросить фильтры", on_click=reset_filters, width="stretch")
        st.divider()
        st.caption(f"Данные: {minimum:%d.%m.%Y} — {maximum:%d.%m.%Y}")
        st.caption("Источник: orders.csv, users.csv, items.csv")
        st.caption("Все показатели, графики, выводы и выгрузка учитывают выбранные фильтры.")

    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Выберите начало и конец периода в боковой панели.")
        st.stop()
    start, end = selected_dates
    selected = filter_data(frame, start, end, categories)
    st.caption(f"{start:%d.%m.%Y} — {end:%d.%m.%Y} · Категорий: {len(categories)} из {len(all_categories)} · Валюта: рубли")
    if len(report["rejected"]) or sum(report["duplicates"].values()) or report["unmatched_items"] or report["unmatched_users"]:
        st.warning("При подготовке обнаружены особенности данных. Подробности — в разделе «Сырые данные».")
    overview, charts, insights, raw = st.tabs([
        "Ключевые показатели", "Визуализация", "Аналитические выводы", "Сырые данные"
    ])
    values = metrics(selected)
    products, cat_revenue, weekdays, daily = aggregate_data(selected, start, end)

    with overview:
        columns = st.columns(4)
        for column, label, value, help_text in zip(columns,
            ["Заказов", "Выручка", "Покупателей", "Средний чек"],
            [number(values["orders"]), rubles(values["revenue"]), number(values["users"]), rubles(values["average_order"])],
            ["Уникальные order_id с выбранными позициями.", "Сумма quantity × price_per_unit. Доход от продаж, без вычета затрат.",
             "Уникальные непустые user_id среди выбранных заказов.", "Выручка выбранных позиций / число заказов с этими позициями."]):
            column.metric(label, value, help=help_text)
        st.caption("При фильтре по категории выручка и средний чек учитывают только выбранные позиции заказов.")
        if selected.empty:
            st.info("Заказов с такими фильтрами нет. Измените период или выберите хотя бы одну категорию.")
        else:
            st.subheader("Выручка по дням")
            fig, ax = figure((11, 3.5))
            x = list(range(len(daily)))
            ax.bar(x, daily["revenue_kopecks"] / 100, color=BLUE, width=0.65)
            # Не более 10 подписей, включая первую и последнюю дату периода.
            ticks = sorted(set(np.linspace(0, len(daily) - 1, min(10, len(daily)), dtype=int)))
            ax.set_xticks(ticks, [daily.index[i].strftime("%d.%m") for i in ticks])
            ax.set_ylabel("Выручка, ₽")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: number(y)))
            ax.grid(axis="y", color=GRID, linewidth=0.7)
            show_figure(fig)

    with charts:
        if selected.empty:
            st.info("Нет данных для графиков. Измените фильтры.")
        else:
            st.subheader("Топ-10 товаров по выручке")
            top = products.head(10)
            barh_chart(top["item_name"].tolist(), (top["revenue_kopecks"] / 100).tolist())
            st.subheader("Выручка по категориям")
            shares = (cat_revenue / cat_revenue.sum() * 100).tolist() if cat_revenue.sum() else None
            barh_chart(cat_revenue.index.tolist(), (cat_revenue / 100).tolist(), shares=shares)
            st.subheader("Заказы по дням недели")
            mode = st.radio("Показатель", ["Всего заказов", "В среднем за календарный день"], horizontal=True)
            average = mode == "В среднем за календарный день"
            plot_values = weekdays["average"] if average else weekdays["orders"]
            fig, ax = figure((11, 3.8))
            bars = ax.bar(range(7), plot_values, color=BLUE, width=0.6)
            labels = [number(v, 2 if average else 0) if pd.notna(v) else "Нет дней" for v in plot_values]
            ax.bar_label(bars, labels=labels, padding=5, color=INK)
            ax.set_xticks(range(7), [f"{day}\nдней: {weekdays.loc[i, 'days']}" for i, day in enumerate(WEEKDAYS)])
            ax.set_ylabel("Заказов в среднем" if average else "Заказов")
            ax.set_ylim(0, max(1, plot_values.max(skipna=True) * 1.22))
            if not average:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="y", color=GRID, linewidth=0.7)
            show_figure(fig)
            st.caption("Число календарных дней указано под столбцами. Среднее = заказы / число таких дней в выбранном периоде, включая дни без заказов.")

    with insights:
        st.subheader("Что показывают данные")
        for finding in conclusions(selected, start, end):
            st.markdown(finding)
        if selected.empty:
            st.info("Для этой выборки выводы недоступны: заказов нет.")
        st.subheader("Границы интерпретации")
        st.markdown(
            f"В исходных данных только **{minimum:%d.%m.%Y} — {maximum:%d.%m.%Y}**. "
            "Результаты относятся к этому фрагменту и выбранным фильтрам; они не являются итогами полного месяца. "
            "Данных о затратах, возвратах, статусах заказов, посещениях и рекламных расходах нет. "
            "Поэтому здесь не рассчитываются прибыль, конверсия и окупаемость рекламы. "
            "`base_price` — справочное поле; выручка рассчитана по фактической цене `price_per_unit`."
        )

    with raw:
        st.subheader("Объединённые данные")
        st.caption(f"Выбрано {len(selected)} из {len(frame)} товарных позиций. Данные после очистки, с текущими фильтрами.")
        view_columns = REQUIRED["orders"] + REQUIRED["users"][1:] + REQUIRED["items"][1:] + ["revenue"]
        view = selected[view_columns]
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            "order_date": st.column_config.DatetimeColumn("order_date", format="DD.MM.YYYY HH:mm"),
            "registration_date": st.column_config.DateColumn("registration_date", format="DD.MM.YYYY"),
            "price_per_unit": st.column_config.NumberColumn("price_per_unit, ₽", format="%.2f"),
            "base_price": st.column_config.NumberColumn("base_price, ₽", format="%.2f"),
            "revenue": st.column_config.NumberColumn("revenue, ₽", format="%.2f"),
        })
        st.download_button("Скачать текущую выборку CSV", view.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"orders_{start}_{end}.csv", mime="text/csv", key="download")
        with st.expander("Исходные таблицы без фильтров"):
            source_name = st.selectbox("CSV-файл", list(sources))
            st.dataframe(sources[source_name], hide_index=True, width="stretch")
        with st.expander("Качество данных и правила расчёта"):
            show_quality(sources, frame, report)
            st.markdown(
                "**Формулы.** Заказы = `order_id.nunique()`, покупатели = `user_id.nunique()`. "
                "Выручка позиции = `quantity × price_per_unit`, общая выручка = сумма позиций. "
                "Средний чек = выручка / число заказов; для пустой выборки — 0. "
                "Денежные суммы складываются в целых копейках, цена округляется до двух знаков. "
                "Количество — целое положительное, цена — неотрицательная. "
                "Сортировка топа: выручка по убыванию, название по алфавиту при равенстве."
            )


if __name__ == "__main__":
    main()
