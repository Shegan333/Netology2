"""Проверки расчётов и поведения фильтров: python -m unittest discover -s tests -v."""

import csv
from datetime import date
from decimal import Decimal
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

import app


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {name: pd.read_csv(app.ROOT / f"{name}.csv") for name in app.REQUIRED}

    def sources_copy(self):
        return {name: frame.copy() for name, frame in self.sources.items()}

    def test_totals_against_independent_decimal_calculation(self):
        with (app.ROOT / "orders.csv").open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        expected = sum(Decimal(row["price_per_unit"]) * int(row["quantity"]) for row in rows)
        frame, report = app.prepare_data(self.sources_copy())
        values = app.metrics(frame)
        self.assertEqual(expected, Decimal("97541.94"))
        self.assertEqual(Decimal(str(values["revenue"])), expected)
        self.assertEqual(values["orders"], 40)
        self.assertEqual(values["users"], 25)
        self.assertAlmostEqual(values["average_order"], 2438.5485)
        self.assertEqual(len(report["rejected"]), 0)
        self.assertEqual(report["unmatched_items"], 0)
        self.assertEqual(report["unmatched_users"], 0)

    def test_aggregates_reconcile_and_known_leader(self):
        frame, _ = app.prepare_data(self.sources_copy())
        products, categories, weekdays, daily = app.aggregate_data(frame, date(2024, 2, 1), date(2024, 2, 10))
        self.assertEqual(int(products.iloc[0]["item_id"]), 102)
        self.assertEqual(products.iloc[0]["revenue_kopecks"], 1499500)
        self.assertEqual(categories["Электроника"], 5326994)
        self.assertEqual(categories.sum(), frame["revenue_kopecks"].sum())
        self.assertEqual(products["revenue_kopecks"].sum(), categories.sum())
        self.assertEqual(daily["revenue_kopecks"].sum(), categories.sum())
        self.assertEqual(weekdays["orders"].sum(), 40)
        self.assertEqual(weekdays["days"].sum(), 10)
        self.assertEqual(weekdays.loc[4, "average"], 4.5)  # 9 заказов / две пятницы

    def test_multi_item_order_is_counted_once(self):
        sources = self.sources_copy()
        extra = sources["orders"].iloc[[0]].copy()
        extra["item_id"], extra["quantity"], extra["price_per_unit"] = 102, 1, 10.0
        sources["orders"] = pd.concat([sources["orders"], extra], ignore_index=True)
        frame, _ = app.prepare_data(sources)
        values = app.metrics(frame)
        self.assertEqual(values["orders"], 40)
        self.assertEqual(values["revenue"], 97551.94)
        self.assertAlmostEqual(values["average_order"], 97551.94 / 40)

    def test_invalid_rows_quarantined_and_unknown_references_kept(self):
        sources = self.sources_copy()
        sources["orders"].loc[0, "order_date"] = "not-a-date"
        sources["orders"].loc[1, "quantity"] = -1
        sources["orders"].loc[2, "price_per_unit"] = float("nan")
        sources["orders"].loc[3, ["user_id", "item_id"]] = [999, 999]
        frame, report = app.prepare_data(sources)
        self.assertEqual(len(report["rejected"]), 3)
        self.assertEqual(len(frame), 37)
        unknown = frame.loc[frame["order_id"].eq(1004)].iloc[0]
        self.assertEqual(unknown["category"], "Неизвестно")
        self.assertEqual(unknown["item_name"], "Товар #999")
        self.assertEqual(unknown["revenue_kopecks"], 349900)
        self.assertEqual(report["unmatched_items"], 1)

    def test_exact_duplicates_removed_and_conflicting_dimension_rejected(self):
        sources = self.sources_copy()
        sources["orders"] = pd.concat([sources["orders"], sources["orders"].iloc[[0]]])
        frame, report = app.prepare_data(sources)
        self.assertEqual(len(frame), 40)
        self.assertEqual(report["duplicates"]["orders"], 1)
        conflict = sources["items"].iloc[[0]].copy()
        conflict["category"] = "Другая категория"
        sources["items"] = pd.concat([sources["items"], conflict])
        with self.assertRaisesRegex(ValueError, "разные записи"):
            app.prepare_data(sources)

    def test_filters_include_end_of_day_and_empty_selection(self):
        frame, _ = app.prepare_data(self.sources_copy())
        selected = app.filter_data(frame, date(2024, 2, 10), date(2024, 2, 10), frame["category"].unique())
        self.assertEqual(app.metrics(selected)["orders"], 3)
        self.assertEqual(selected["order_date"].max().hour, 15)
        empty = app.filter_data(frame, date(2024, 2, 1), date(2024, 2, 10), [])
        self.assertEqual(app.metrics(empty), {"orders": 0, "revenue": 0.0, "users": 0, "average_order": 0.0})
        self.assertEqual(app.conclusions(empty, date(2024, 2, 1), date(2024, 2, 10)), [])

    def test_all_invalid_data_and_missing_schema(self):
        sources = self.sources_copy()
        sources["orders"]["price_per_unit"] = -1
        frame, report = app.prepare_data(sources)
        self.assertTrue(frame.empty)
        self.assertEqual(len(report["rejected"]), 40)
        sources["orders"].drop(columns=["quantity"], inplace=True)
        with self.assertRaisesRegex(ValueError, "quantity"):
            app.prepare_data(sources)

    def test_zero_revenue_and_tied_unit_leaders(self):
        sources = self.sources_copy()
        frame, _ = app.prepare_data(sources)
        findings = app.conclusions(frame, date(2024, 2, 1), date(2024, 2, 10))
        unit_text = next(text for text in findings if "Лидеры по проданным" in text)
        for name in ["Смартфон Xiaomi Redmi", "Футболка хлопковая", 'Книга "Python для анализа данных"', "Кофеварка De'Longhi"]:
            self.assertIn(name, unit_text)
        sources["orders"]["price_per_unit"] = 0
        zero, _ = app.prepare_data(sources)
        self.assertEqual(app.metrics(zero)["revenue"], 0)
        self.assertIn("равна нулю", app.conclusions(zero, date(2024, 2, 1), date(2024, 2, 10))[0])


class InterfaceTests(unittest.TestCase):
    def test_default_category_empty_and_reset(self):
        ui = AppTest.from_file(str(app.ROOT / "app.py"), default_timeout=40).run()
        self.assertFalse(ui.exception)
        self.assertEqual([metric.value for metric in ui.metric], ["40", "97 541,94 ₽", "25", "2 438,55 ₽"])
        ui.multiselect(key="categories").set_value(["Электроника"]).run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.metric[1].value, "53 269,94 ₽")
        self.assertTrue(any("Электроника" in value.value for value in ui.markdown))
        ui.multiselect(key="categories").set_value([]).run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.metric[0].value, "0")
        self.assertTrue(ui.info)
        ui.button[0].click().run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.metric[0].value, "40")
        ui.date_input(key="period").set_value((date(2024, 2, 10), date(2024, 2, 10))).run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.metric[0].value, "3")
        ui.radio[0].set_value("В среднем за календарный день").run()
        self.assertFalse(ui.exception)


if __name__ == "__main__":
    unittest.main()
