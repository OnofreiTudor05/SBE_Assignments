import numpy as np
import operator
import random

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
        new_publication.station_id = self.station_pool[np.random.randint(0, len(self.station_pool))]
        new_publication.city = self.city_pool[np.random.randint(0, len(self.city_pool))]
        new_publication.direction = self.direction_pool[np.random.randint(0, len(self.direction_pool))]
        new_publication.date = self.date_pool[np.random.randint(0, len(self.date_pool))]
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
        self.requried_value = required_value

    def __str__(self) -> str:
        return f"(factor: {self.factor}, operator: {self.operator}, value: {self.requried_value});"

class Subscription:
    def __init__(self, constraints=None) -> None:
        self.constraints = constraints

    def __str__(self) -> str:
        return self.constraints
    
    def is_constraint_respected(self, position, publication):
        current_constraint = self.constraints[position]
        current_operation = operator_dict.get(current_constraint.operator)
        evaluation_string = "current_operation(getattr(publication, current_constraint.factor), current_constraint.required_value)"
        return eval(evaluation_string)
    
class SubscriptionGenerator:
    def __init__(self, publication_generator = None, required_weights = None) -> None:
        self.publication_generator = publication_generator
        # list of tuples [(0.3, 'city'), (0.3, 'date'), (0.2, 'station'), (0.2, 'wind')]
        self.required_weights = required_weights

        count_city = int(subscriptions_count * freq_city)
        remaining_count = (subscriptions_count - count_city) // (len(vars(Publication())) - 1)
        
        chosen_field = random.choice(len(vars(Publication())))
        field_count = int(subscriptions_count * eq_op_freq)
        curr_count_eq_op = 0
        
        output_subscriptions = []
        for i in range(subscriptions_count):
            iterator_subscription = []
            for key, value in vars((Publication()).items():
                match key:
                    case "city":
                        iterator_subscription.append((key, "==" if random.random() < freq_city else "!=", cities[np.random.randomint(0, len(cities))]))
                    case "directions":
                        iterator_subscription.append((key, "==" if random.randint(1, 11) % 2 == 0 else "!=", cities[np.random.randomint(0, len(directions))]))
                    case "dates":
                        iterator_subscription.append((key, "==" if random.randint(1, 11) % 2 == 0 else "!=", cities[np.random.randomint(0, len(directions))]))
                    case _:
                        
                    
                                   
                                   
if __name__ == "__main__":
    generator = PublicationGenerator(stations, cities, directions, dates, temp_limits, wind_limits, rain_limits)
    publications = generator.generate_publications(5)
    print(*[f"{str(x)}" for x in publications], sep='\n')
    
    subscription_generator = SubscriptionGenerator().generate_subscription(1000, 10000, 0.9, 0.7)
    print(subscription_generator)