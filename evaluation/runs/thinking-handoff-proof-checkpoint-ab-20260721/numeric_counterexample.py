#!/usr/bin/env python3
"""Check the circumcenters claimed by the two generated proofs."""

from __future__ import annotations

import math

Point = tuple[float, float]


def add(left: Point, right: Point) -> Point:
    return left[0] + right[0], left[1] + right[1]


def subtract(left: Point, right: Point) -> Point:
    return left[0] - right[0], left[1] - right[1]


def scale(value: float, point: Point) -> Point:
    return value * point[0], value * point[1]


def dot(left: Point, right: Point) -> float:
    return left[0] * right[0] + left[1] * right[1]


def norm(point: Point) -> float:
    return math.sqrt(dot(point, point))


def distance(left: Point, right: Point) -> float:
    return norm(subtract(left, right))


def second_circle_intersection(
    first: Point, direction: Point, center: Point
) -> Point:
    relative = subtract(first, center)
    parameter = -2.0 * dot(relative, direction) / dot(direction, direction)
    return add(first, scale(parameter, direction))


def circumcenter(first: Point, second: Point, third: Point) -> Point:
    a11 = 2.0 * (second[0] - first[0])
    a12 = 2.0 * (second[1] - first[1])
    a21 = 2.0 * (third[0] - first[0])
    a22 = 2.0 * (third[1] - first[1])
    b1 = dot(second, second) - dot(first, first)
    b2 = dot(third, third) - dot(first, first)
    determinant = a11 * a22 - a12 * a21
    return (
        (b1 * a22 - a12 * b2) / determinant,
        (a11 * b2 - b1 * a21) / determinant,
    )


def point_line_distance(point: Point, anchor: Point, direction: Point) -> float:
    offset = subtract(point, anchor)
    cross = offset[0] * direction[1] - offset[1] * direction[0]
    return abs(cross) / norm(direction)


def format_point(point: Point) -> str:
    return f"({point[0]:.9f}, {point[1]:.9f})"


def main() -> None:
    d = 1.0
    r = 1.0
    radius_gamma = 1.5

    h = (d * d + r * r - radius_gamma * radius_gamma) / (2.0 * d)
    a = math.sqrt(r * r - h * h)
    point_a = (h, a)
    point_b = (h, -a)
    point_p = (
        (d + radius_gamma - r) / 2.0,
        -((d + radius_gamma - r) / 2.0) * (h + r) / a,
    )
    direction_ap = subtract(point_p, point_a)
    point_e = second_circle_intersection(point_a, direction_ap, (0.0, 0.0))
    point_f = second_circle_intersection(point_a, direction_ap, (d, 0.0))
    point_h = (
        point_p[0],
        point_p[0] * (d - point_p[0]) / point_p[1],
    )

    actual_center = circumcenter(point_b, point_e, point_f)
    actual_distances = [
        distance(actual_center, point) for point in (point_b, point_e, point_f)
    ]
    radius = actual_distances[0]
    tangent_distance = point_line_distance(actual_center, point_h, direction_ap)

    u = math.sqrt(4.0 - radius_gamma * radius_gamma)
    denominator = 3.0 * radius_gamma**2 - 8.0 * radius_gamma + 8.0
    checkpoint_center = (0.5, -8.0 / (u * denominator))

    x_value = d + radius_gamma - r
    lossless_term = (x_value + 2.0) * (radius_gamma - r) / (2.0 * a)
    lossless_center = (h + lossless_term, -a + lossless_term)

    print(f"A = {format_point(point_a)}")
    print(f"B = {format_point(point_b)}")
    print(f"P = {format_point(point_p)}")
    print(f"E = {format_point(point_e)}")
    print(f"F = {format_point(point_f)}")
    print(f"H = {format_point(point_h)}")
    print(f"actual circumcenter = {format_point(actual_center)}")
    print(f"actual B/E/F distances = {[round(value, 9) for value in actual_distances]}")
    print(f"actual radius = {radius:.9f}")
    print(f"distance from actual center to claimed tangent line = {tangent_distance:.9f}")

    for name, center in (
        ("proof_checkpoint", checkpoint_center),
        ("lossless_partial", lossless_center),
    ):
        distances = [distance(center, point) for point in (point_b, point_e, point_f)]
        print(f"{name} center = {format_point(center)}")
        print(f"{name} B/E/F distances = {[round(value, 9) for value in distances]}")
        assert max(distances) - min(distances) > 0.1

    assert max(actual_distances) - min(actual_distances) < 1e-9
    assert math.isclose(radius, tangent_distance, rel_tol=0.0, abs_tol=1e-9)


if __name__ == "__main__":
    main()
