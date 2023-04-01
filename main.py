import numpy as np
import operator
import random
import logging

STATIONS = [1, 4, 7, 15, 23, 27, 69, 100]
CITIES = ["Bucharest", "Harlau", "Braila", "Galati", "Darabani", "Dubai"]
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DATES = [f"{i}.04.2023" for i in range(10, 17)]
TEMP_LIMITS = (-20, 40)
WIND_LIMITS = (0, 100)
RAIN_LIMITS = (0, 1)

FIELDS = ["stationId", "city", "direction", "date", "temp", "wind", "rain"]
OPERATORS = ["==", "!=", "<", ">", "<=", ">="]
VALID_OPERATIONS = {
    "stationId": ["==", "!=", "<", ">", "<=", ">="],
    "city": ["=="],
    "direction": ["=="],
    "date": ["=="],
    "temp": ["==", "!=", "<", ">", "<=", ">="],
    "wind": ["==", "!=", "<", ">", "<=", ">="],
    "rain": ["==", "!=", "<", ">", "<=", ">="]
}

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
    
class SubscriptionGeneratorV2:
    def __init__(self, publication_generator=None, subscription_count=None, required_field_weights=None, required_operator_weights=None) -> None:
        self.publication_generator = publication_generator
        self.subscription_count = subscription_count
        self.required_field_weights = self.validate_field_weights(required_field_weights)
        self.required_operator_weights = self.validate_operator_weights(required_operator_weights)
        self.total_validation()

    def validate_field_weights(self, field_weights):
        """Method used to filter and validate field weights provided
        to the constructor. If a weight is not valid, it will log
        an error and stop.

        Args:
            field_weights (list): a list containing tuples, the first 
            element in the tuple is the field (string) and the second
            is a weight (a float value 0.0 < x < 1.0).

        Returns:
            list: the same list of tuples which is validated.
        """
        for fw in field_weights:
            if fw[1] < 0 or fw[1] > 1:
                logging.error(f"Field {fw[0]} doesn't use a valid weight.")
                exit(1)
        return field_weights

    def validate_operator_weights(self, operator_weights):
        """Method used to filter and validate operator weights provided
        to the constructor. If a weight is not valid, it will log
        an error and stop.

        Args:
            operator_weights (list): a list containing tuples, the first 
            element in the tuple is the operator (string) and the second
            is a weight (a float value 0.0 < x < 1.0).

        Returns:
            list: the same list of tuples which is validated.
        """
        for ow in operator_weights:
            if ow[0] not in OPERATORS:
                logging.error(f"Operator {ow[0]} is not valid.")
                exit(1)
            if ow[1] < 0 or ow[1] > 1:
                logging.error(f"Operator {ow[1]} doesn't use a valid weight.")
                exit(1)
        return operator_weights
    
    def total_validation(self):
        """Method used as a final validation in the constructor.
        It takes the field and operator and checks an existing map
        of valid operators to check if it can be used for the specified
        field. It will log an error and exit the program if the validation
        condition is not met.
        """
        # Short-circuit
        if len(self.required_field_weights) != len(self.required_operator_weights):
            logging.error("Validation failed due two inconsistent number of field and operator weights provided.")
        mapped_elements = list(map(lambda t: (t[0][0], t[1][0]), zip(self.required_field_weights, self.required_operator_weights)))
        for me in mapped_elements:
            if me[1] not in VALID_OPERATIONS[me[0]]:
                logging.error(f"Invalid operator {me[1]}. It doesn't correspond with the available operators for the {me[0]} field.")
                exit(1)

    def generate_subscriptions(self):
        # 0 field name 1 field weight 2 operator 3 operator weight
        mapped_field_operator_weights = [t1 + t2 for t1, t2 in zip(self.required_field_weights, self.required_operator_weights)]
        mapped_field_operator_weights = sorted(mapped_field_operator_weights, key=lambda x: x[1], reverse=True)

        combined_constraints_list = list()
        for mfow in mapped_field_operator_weights:
            constraints_list = list()
            field_subscription_count = int(mfow[1] * self.subscription_count)

            if field_subscription_count + int((1 - mfow[1]) * self.subscription_count) != self.subscription_count:
                logging.warning(f"A total of {self.subscription_count - field_subscription_count - int((1 - mfow[1]) * self.subscription_count)} subscriptions have been lost due to rounding.")

            operator_field_count = int(mfow[3] * field_subscription_count)

            if operator_field_count + int((1 - mfow[3]) * field_subscription_count) != field_subscription_count:
                logging.warning(f"A total of {field_subscription_count - operator_field_count - int((1 - mfow[3]) * field_subscription_count)} correct matches have been lost due to rounding.")

            current_operator_count = 0
            for _i in range(0, field_subscription_count):
                match mfow[0]:
                    case "city":
                        constraint = Constraint(
                            mfow[0],
                            mfow[2] if current_operator_count < operator_field_count else "!=",
                            CITIES[np.random.randint(0, len(CITIES) - 1)]
                        )
                        if constraint.operator == mfow[2]:
                            current_operator_count += 1
                        constraints_list.append(constraint)
                    case "stationId":
                        constraint = Constraint(
                            mfow[0],
                            mfow[2] if current_operator_count < operator_field_count else np.random.choice(VALID_OPERATIONS["stationId"].remove(mfow[2])),
                            STATIONS[np.random.randint(0, len(STATIONS) - 1)]
                        )
                        if constraint.operator == mfow[2]:
                            current_operator_count += 1
                        constraints_list.append(constraint)
                    case "direction":
                        constraint = Constraint(
                            mfow[0],
                            mfow[2] if current_operator_count < operator_field_count else "!=",
                            DIRECTIONS[np.random.randint(0, len(DIRECTIONS) - 1)]
                        )
                        if constraint.operator == mfow[2]:
                            current_operator_count += 1
                        constraints_list.append(constraint)
                    case "date":
                        constraint = Constraint(
                            mfow[0],
                            mfow[2] if current_operator_count < operator_field_count else "!=",
                            DATES[np.random.randint(0, len(DATES) - 1)]
                        )
                        if constraint.operator == mfow[2]:
                            current_operator_count += 1
                        constraints_list.append(constraint)
                    case "temp":
                        constraint = Constraint(
                            mfow[0],
                            mfow[2] if current_operator_count < operator_field_count else np.random.choice(VALID_OPERATIONS["temp"].remove(mfow[2])),
                            np.random.randint(TEMP_LIMITS[0], TEMP_LIMITS[1])
                        )
                        if constraint.operator == mfow[2]:
                            current_operator_count += 1
                        constraints_list.append(constraint)
                    case "wind":
                        constraint = Constraint(
                            mfow[0],
                            mfow[2] if current_operator_count < operator_field_count else np.random.choice(VALID_OPERATIONS["wind"].remove(mfow[2])),
                            np.random.randint(WIND_LIMITS[0], WIND_LIMITS[1])
                        )
                        if constraint.operator == mfow[2]:
                            current_operator_count += 1
                        constraints_list.append(constraint)
                    case "rain":
                        constraint = Constraint(
                            mfow[0],
                            mfow[2] if current_operator_count < operator_field_count else np.random.choice(VALID_OPERATIONS["rain"].remove(mfow[2])),
                            round(np.random.uniform(RAIN_LIMITS[0], RAIN_LIMITS[1]), 1)
                        )
                        if constraint.operator == mfow[2]:
                            current_operator_count += 1
                        constraints_list.append(constraint)
                    case _:
                        logging.error("Argument was not matched.")
            combined_constraints_list.append(constraints_list)
        return combined_constraints_list

    def compress_generated_constraints(self):
        pass

if __name__ == "__main__":
    generator = PublicationGenerator(STATIONS, CITIES, DIRECTIONS, DATES, TEMP_LIMITS, WIND_LIMITS, RAIN_LIMITS)
    publications = generator.generate_publications(5)
    print(*[f"{str(x)}" for x in publications], sep='\n')
    
    # a list of tuples [(field, frequency), ...]
    field_weights = [
        ("stationId", 0.7),
        ("city", 0.5),
        ("direction", 0.8),
        ("date", 0.2),
        ("temp", 0.1),
        ("wind", 0.3),
        ("rain", 0.5)
    ]

    operator_weights = [        
        ("==", 0.5),
        ("==", 0.9),
        ("==", 0.5),
        ("==", 0.2),
        (">", 0.8),
        (">=", 0.3),
        ("<", 0.3)
    ]

    sub_generator = SubscriptionGeneratorV2(generator, 100, field_weights, operator_weights)
    constraints = sub_generator.generate_subscriptions()
    for constraint in constraints:
        for c in constraint:
            print(str(c))