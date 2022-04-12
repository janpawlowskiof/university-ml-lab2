from dataclasses import dataclass

import numpy as np
import scipy.spatial


@dataclass
class Cities:
    coordinates: np.ndarray
    demand: np.ndarray

    def __len__(self):
        return len(self.coordinates)


class Config:
    capacity: int
    cities: Cities
    distance_matrix: np.ndarray

    @staticmethod
    def _calculate_distance_matrix():
        return scipy.spatial.distance_matrix(Config.cities.coordinates, Config.cities.coordinates).astype(np.float32)

    @staticmethod
    def num_cities() -> int:
        return len(Config.cities)

    @staticmethod
    def from_path(path):
        with path.open("r", encoding="utf8") as file:
            lines = file.readlines()

            capacity = int(list(filter(lambda line: line.startswith("CAPACITY"), lines))[0].split(":")[-1])
            num_cities = int(list(filter(lambda line: line.startswith("DIMENSION"), lines))[0].split(":")[-1])

            coordinates_start_index = Config._get_index_of_line(lines, "NODE_COORD_SECTION") + 1
            coordinate_lines = lines[coordinates_start_index:coordinates_start_index + num_cities]
            coordinate_lines = [map(int, coordinate_line.split(" ")) for coordinate_line in coordinate_lines]
            coordinates = np.stack([
                np.array([x, y])
                for index, x, y
                in coordinate_lines
            ])

            demand_start_index = Config._get_index_of_line(lines, "DEMAND_SECTION") + 1
            demand_lines = lines[demand_start_index:demand_start_index + num_cities]
            demand_lines = [map(int, demand_line.split(" ")) for demand_line in demand_lines]
            demands = np.stack([
                demand
                for index, demand
                in demand_lines
            ])

            cities = Cities(coordinates, demands)

            Config.capacity = capacity
            Config.cities = cities
            Config.distance_matrix = Config._calculate_distance_matrix()

    @staticmethod
    def _get_index_of_line(lines, line_content):
        return [line_index for line_index, line in enumerate(lines) if line_content in line][0]
