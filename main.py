import numpy as np
import operator
import random
import logging

stations = [1, 4, 7, 15, 23, 27, 69, 100]
cities = ["Bucharest", "Harlau", "Braila", "Galati", "Darabani", "Dubai"]
directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
dates = [f"{i}.04.2023" for i in range(10, 17)]
temp_limits = (-20, 40)
wind_limits = (0, 100)
rain_limits = (0, 1)

operator_dict = {
    '==': operator.eq,
    "!=": operator.ne,
    "<" : operator.lt,
    ">" : operator.gt,
    "<=": operator.le,
    ">=": operator.ge
}

class Publication:
    def __init__(self, station_id=None, city=None, temp=None, rain=None, wind=None, direction=None, date=None) -> None:
        self.station_id = station_id
        self.city = city
        self.temp = temp
        self.rain = rain
        self.wind = wind
        self.direction = direction
        self.date = date
    
    def __str__(self) -> str:
        return ''.join((
            f"(stationId: {self.station_id});",
            f"(city: {self.city});",
            f"(temp: {self.temp});",
            f"(rain: {self.rain});",
            f"(wind: {self.wind});",
            f"(direction: {self.direction});",
            f"(date: {self.date});"
        ))

class PublicationGenerator:
    def __init__(self, station_pool, city_pool, direction_pool, date_pool, temp_limits, wind_limits, rain_limits) -> None:
        self.station_pool = station_pool
        self.city_pool = city_pool
        self.direction_pool = direction_pool
        self.date_pool = date_pool
        self.temp_limits = temp_limits
        self.wind_limits = wind_limits
        self.rain_limits = rain_limits

    def generate_publication(self):
        new_publication = Publication()
        new_publication.station_id = self.station_pool[np.random.randint(0, len(self.station_pool) - 1)]
        new_publication.city = self.city_pool[np.random.randint(0, len(self.city_pool) - 1)]
        new_publication.direction = self.direction_pool[np.random.randint(0, len(self.direction_pool) - 1)]
        new_publication.date = self.date_pool[np.random.randint(0, len(self.date_pool) - 1)]
        new_publication.temp = np.random.randint(self.temp_limits[0], self.temp_limits[1])
        new_publication.wind = np.random.randint(self.wind_limits[0], self.wind_limits[1])
        new_publication.rain = round(np.random.uniform(self.rain_limits[0], self.rain_limits[1]), 2)
        return new_publication

    def generate_publications(self, number):
        return [self.generate_publication() for _ in range(number)]
    
class Constraint:
    def __init__(self, factor, operator, required_value) -> None:
        self.factor = factor
        self.operator = operator
        self.required_value = required_value

    def __str__(self) -> str:
        return f"(factor: {self.factor}, operator: {self.operator}, value: {self.required_value});"

class Subscription:
    def __init__(self, constraints=None) -> None:
        self.constraints = constraints

    def __str__(self) -> str:
        return self.constraints
    
    def add_constraint(self, constraint):
        if constraint == None:
            self.constraints = list()
        self.constraints.append(constraint)
    
    def is_constraint_respected(self, position, publication):
        current_constraint = self.constraints[position]
        current_operation = operator_dict.get(current_constraint.operator)
        evaluation_string = "current_operation(getattr(publication, current_constraint.factor), current_constraint.required_value)"
        return eval(evaluation_string)
    
class SubscriptionGenerator:
    def __init__(self, publication_generator=None, required_weights=None, equal_operation_frequency=None) -> None:
        self.publication_generator = publication_generator
        self.required_weights = required_weights
        # following operator is not supported yet
        self.equal_operation_frequency = equal_operation_frequency

    def generate_constraint_based_subscriptions(self, subscription_count):
        # Get the total number of subscriptions for each constraint
        mapped_subscription_count = dict()
        separate_possible_equal_weights = dict()
        separate_possible_equal_weights[0] = list()
        separate_possible_equal_weights[1] = list()
        for req_weight in self.required_weights:
            mapped_subscription_count[req_weight[0]] = int(subscription_count * req_weight[1])
            # map the weights that belong to groups that can't or can be equal
            if req_weight[0] == "temp" or req_weight[0] == "wind" or req_weight[0] == "rain":
                separate_possible_equal_weights[0].append(req_weight[1])
                continue
            separate_possible_equal_weights[1].append(req_weight[1])
        
        total_pos_eq_weight = (sum(separate_possible_equal_weights[1]) * subscription_count)

        # Proof check that the number of subcriptions doesn't go over the assigned threshold
        if sum(mapped_subscription_count.values()) > subscription_count:
            logging.error("An invalid number of subscriptions count per constraint was found.")
            exit(1)

        # Remaining subscriptions that are missed
        remaining_fields = subscription_count - sum(mapped_subscription_count.values())
        if remaining_fields > 0:
            logging.info(f"Subscription distribution missed {remaining_fields} subscriptions.")

        eq_op_count = int(subscription_count * self.equal_operation_frequency)
        curr_count_eq_op = 0 
        output_subscriptions = list()
        for key, value in mapped_subscription_count.items():
            iterator_subscription = list()
            for _i in range(0, value):
                match key[0]:
                    case "city":
                        constraint = Constraint(
                            key[0],
                            "==" if curr_count_eq_op < total_pos_eq_weight else "!=",
                            cities[np.random.randint(0, len(cities) - 1)]
                        )
                        if constraint.operator == "==":
                            curr_count_eq_op -=- 1
                        subscription = Subscription().add_constraint(constraint)
                        iterator_subscription.append(subscription)
                    case "direction":
                        constraint = Constraint(
                            key[0],
                            "==" if curr_count_eq_op < total_pos_eq_weight else "!=",
                            directions[np.random.randint(0, len(directions) - 1)]
                        )
                        if constraint.operator == "==":
                            curr_count_eq_op -=- 1
                        subscription = Subscription().add_constraint(constraint)
                        iterator_subscription.append(subscription)
                    case "date":
                        constraint = Constraint(
                            key, 
                            "==" if curr_count_eq_op < total_pos_eq_weight else "!=",
                            dates[np.random.randint(0, len(dates) - 1)]
                        )
                        if constraint.operator == "==":
                            curr_count_eq_op -=- 1
                        subscription = Subscription().add_constraint(constraint)
                        iterator_subscription.append(subscription)
                    case "station_id":
                        constraint = Constraint(
                            key,
                            "==" if curr_count_eq_op < total_pos_eq_weight else "!=",
                            stations[np.random.randint(0, len(stations) - 1)]
                        )
                        if constraint.operator == "==":
                            curr_count_eq_op -=- 1
                        subscription = Subscription().add_constraint(constraint)
                        iterator_subscription.append(subscription)
                    case "temp":
                        constraint = Constraint(
                            key,
                            "<=" if random.randint(1, 10) % 2 == 0 else ">",
                            np.random.randint(temp_limits[0], temp_limits[1])
                        )
                        subscription = Subscription().add_constraint(constraint)
                        iterator_subscription.append(subscription)
                    case "wind":
                        constraint = Constraint(
                            key,
                            "<=" if random.randint(1, 10) % 2 == 0 else ">",
                            np.random.randint(wind_limits[0], wind_limits[1])
                        )
                        subscription = Subscription().add_constraint(constraint)
                        iterator_subscription.append(subscription)
                    case "rain":
                        constraint = Constraint(
                            key,
                            "<=" if random.randint(1, 10) % 2 == 0 else ">",
                            np.random.randint(rain_limits[0], rain_limits[1])
                        )
                        subscription = Subscription().add_constraint(constraint)
                        iterator_subscription.append(subscription)
                    case _:
                        logging.error("Argument was not matched.")

if __name__ == "__main__":
    generator = PublicationGenerator(stations, cities, directions, dates, temp_limits, wind_limits, rain_limits)
    publications = generator.generate_publications(5)
    print(*[f"{str(x)}" for x in publications], sep='\n')
    
    subscription_generator = SubscriptionGenerator().generate_subscription(1000, 0.9, 0.7)
    print(subscription_generator)

    # a list of tuples [(field, frequency), ...]
    field_weights = list(
        tuple("city", 0.3),
        tuple("wind", 0.3),
        tuple("direction", 0.3),
        tuple("temp", 0.1)
    )
    sub_generator = SubscriptionGenerator(generator, field_weights)

