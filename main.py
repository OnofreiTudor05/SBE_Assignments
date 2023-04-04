import numpy as np
import operator
import logging
from configparser import ConfigParser
import threading
from collections import deque

config_object = ConfigParser()
config_object.read("config.ini")

STATIONS = [int(x) for x in config_object["FIELDPROPERTIES"]["stationIds"].split()]
CITIES = config_object["FIELDPROPERTIES"]["cities"].split()
DIRECTIONS = config_object["FIELDPROPERTIES"]["directions"].split()
DATES = config_object["FIELDPROPERTIES"]["dates"].split()
TEMP_LIMITS = tuple(int(x) for x in config_object["FIELDPROPERTIES"]["temp"].split())
WIND_LIMITS = tuple(int(x) for x in config_object["FIELDPROPERTIES"]["wind"].split())
RAIN_LIMITS = tuple(float(x) for x in config_object["FIELDPROPERTIES"]["rain"].split())

FIELDS = config_object["FIELDS"]["fields"].split()
OPERATORS = config_object["FIELDS"]["operators"].split()
VALID_OPERATIONS = {
    "stationId": config_object["FIELDVALIDOPERATORS"]["opstationId"].split(),
    "city": [config_object["FIELDVALIDOPERATORS"]["opcity"]],
    "direction": [config_object["FIELDVALIDOPERATORS"]["opdirection"]],
    "date": [config_object["FIELDVALIDOPERATORS"]["opdate"]],
    "temp": config_object["FIELDVALIDOPERATORS"]["optemp"].split(),
    "wind": config_object["FIELDVALIDOPERATORS"]["opwind"].split(),
    "rain": config_object["FIELDVALIDOPERATORS"]["oprain"].split()
}

field_filter = ["fwstationIds", "fwcities", "fwdirections", "fwdates", "fwtemp", "fwwind", "fwrain"]
FIELD_WEIGHTS = []
for ff in field_filter:
    try:
        FIELD_WEIGHTS.append(tuple(float(val) if "." in val else val for val in config_object["FIELDWEIGHTS"][ff].split()))
    except KeyError:
        pass

operator_filter = ["owstationIds", "owcities", "owdirections", "owdates", "owtemp", "owwind", "owrain"]
OPERATOR_WEIGHTS = []
for of in operator_filter:
    try:
        OPERATOR_WEIGHTS.append(tuple(float(val) if "." in val else val for val in config_object["OPERATORWEIGHTS"][of].split()))
    except KeyError:
        pass

MIN_THREAD_COUNT = int(config_object["THREADSETUP"]["min_count"])
MAX_THREAD_COUNT = int(config_object["THREADSETUP"]["max_count"])
SELECTED_THREAD_COUNT = int(config_object["THREADSETUP"]["selected_count"])

operator_dict = {
    '==': operator.eq,
    "!=": operator.ne,
    "<" : operator.lt,
    ">" : operator.gt,
    "<=": operator.le,
    ">=": operator.ge
}

global publication_list
publication_list = list()

global combined_constraint_list
combined_constraints_list = list()

global subscription_list
subscription_list = list()

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
    
    def generate_publication_thread(self, number_of_publications):
        publication_list.extend([self.generate_publication() for _ in range(number_of_publications)])

    def generate_publications(self, number):
        thread_list = list()
        number_per_thread = int(number / SELECTED_THREAD_COUNT)

        first = 0
        if number_per_thread * SELECTED_THREAD_COUNT < number:
            logging.warning("Adjusting number of publications per thread to be generated to fit the specific number provided.")
            first = number - number_per_thread * SELECTED_THREAD_COUNT

        current_number_per_thread = number_per_thread
        for i in range(0, SELECTED_THREAD_COUNT):
            number_per_thread = (current_number_per_thread + first) if i == 0 and first != 0 else current_number_per_thread
            thread = threading.Thread(target=self.generate_publication_thread, args=(number_per_thread,))
            thread_list.append(thread)

        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()

        return publication_list
    
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
        reprt = "Subscription Constraints: ("
        for constraint in self.constraints:
            reprt += str(constraint)
        return reprt + ")"
    
    def add_constraint(self, constraint):
        if self.constraints == None:
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

    def group_fields_by_weight(self, sorted_map):
        """The method takes a sorted map as input and returns a 
        list of groups of fields, where each group contains fields
        whose weights have a sum or less than 1.0.
        It works by iterating over the sorted map and maintaining a current
        group of fields and a current weight sum. For each field, it checks
        whether adding it's weight to the current weight sum would exceed 1.
        If not, it adds the field to the current group and updates the weight
        sum. If adding the field would exceed 1.0, it starts a new group with
        the current field and weight sum.
        At the end, if there is a non-empty group left, it is added to the list
        of groups.

        Args:
            sorted_map (list): sorted map containing field_weights mapping

        Returns:
            list: groups of field_weights mapped for thread usage
        """
        groups = list()
        current_group = deque()
        current_weight_sum = 0.0
        last_index = -1

        for i, (field_name, field_weight, operator, operator_weight) in enumerate(sorted_map):
            if current_weight_sum + field_weight > 1.0:
                if last_index >= 0:
                    groups.append(list(current_group)[:(last_index + 1)])
                    for _ in  range(last_index + 1):
                        if current_group:
                            current_group.popleft()
                    current_weight_sum -= sum(field_weight for _, field_weight, _, _ in groups[-1])
                    last_index = -1
                else:
                    groups.append([current_group.popleft()])
                    current_weight_sum = 0.0
            current_group.append((field_name, field_weight, operator, operator_weight))
            current_weight_sum += field_weight
            if last_index < 0 and current_weight_sum >= 1.0:
                last_index = i

        if current_group:
            groups.append(list(current_group))

        return groups
    
    def generate_constraints_thread(self, group):
        for mfow in group:
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
                            mfow[2] if current_operator_count < operator_field_count else np.random.choice([x for x in VALID_OPERATIONS["stationId"] if x != mfow[2]]),
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
                            mfow[2] if current_operator_count < operator_field_count else np.random.choice([x for x in VALID_OPERATIONS["temp"] if x != mfow[2]]),
                            np.random.randint(TEMP_LIMITS[0], TEMP_LIMITS[1])
                        )
                        if constraint.operator == mfow[2]:
                            current_operator_count += 1
                        constraints_list.append(constraint)
                    case "wind":
                        constraint = Constraint(
                            mfow[0],
                            mfow[2] if current_operator_count < operator_field_count else np.random.choice([x for x in VALID_OPERATIONS["wind"] if x != mfow[2]]),
                            np.random.randint(WIND_LIMITS[0], WIND_LIMITS[1])
                        )
                        if constraint.operator == mfow[2]:
                            current_operator_count += 1
                        constraints_list.append(constraint)
                    case "rain":
                        constraint = Constraint(
                            mfow[0],
                            mfow[2] if current_operator_count < operator_field_count else np.random.choice([x for x in VALID_OPERATIONS["rain"] if x != mfow[2]]),
                            round(np.random.uniform(RAIN_LIMITS[0], RAIN_LIMITS[1]), 1)
                        )
                        if constraint.operator == mfow[2]:
                            current_operator_count += 1
                        constraints_list.append(constraint)
                    case _:
                        logging.error("Argument was not matched.")
            combined_constraints_list.append(constraints_list)

    def generate_constraints(self):
        """The method takes the field weights and operator weights and
        created a sorted map depending on the weight of the field. Then
        the result is grouped in lists of elements with a summed weight
        less or equal to 1.0.
        Each obtained group is passed to a thread which creates constraints
        based on the weight of the fields and operators.

        Returns:
            list: a list of generated constraints.
        """
        # 0 field name 1 field weight 2 operator 3 operator weight
        mapped_field_operator_weights = [t1 + t2 for t1, t2 in zip(self.required_field_weights, self.required_operator_weights)]
        mapped_field_operator_weights = sorted(mapped_field_operator_weights, key=lambda x: x[1], reverse=True)

        grouped_field_operator_weights = self.group_fields_by_weight(mapped_field_operator_weights)

        thread_list = list()
        for gfow in grouped_field_operator_weights:
            thread = threading.Thread(target=self.generate_constraints_thread, args=(gfow,))
            thread_list.append(thread)

            if len(thread_list) == SELECTED_THREAD_COUNT:
                for thread in thread_list:
                    thread.start()

                for thread in thread_list:
                    thread.join()
                
                thread_list = list()

        # Solve any remaining threads
        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()

        return combined_constraints_list
    
    def compress_generated_constraints(self):
        constraint_pool = self.generate_constraints()
        subscription_list = list()

        for constraint in constraint_pool[0]:
            sub = Subscription()
            sub.add_constraint(constraint)
            subscription_list.append(sub)
        
        for constraint_list in constraint_pool[1:]:
            for position, constraint in enumerate(constraint_list):
                if len(subscription_list) < self.subscription_count:
                    sub = Subscription()
                    sub.add_constraint(constraint)
                    subscription_list.append(sub)
                else:
                    break

            current_position = position
            if current_position < len(constraint_list):
                for subscription in subscription_list:
                    subscription.add_constraint(constraint_list[current_position])
                    current_position += 1
                    if current_position >= len(constraint_list):
                        break

            np.random.shuffle(subscription_list)
        return subscription_list
    
def validate_generated_publications(pub_count, publications):
    if len(publications) != pub_count:
        logging.warning(f"The number of requested publications: {pub_count} does not match the number of publications generated: {len(publications)}")
    
    wrong_matched_count = 0
    for publication in publications:
        if publication.station_id not in STATIONS:
            logging.error(f"The station id {publication.station_id} of the publication doesn't match any provided station id.")
            wrong_matched_count += 1
        if publication.city not in CITIES:
            logging.error(f"The city {publication.city} of the publication doesn't match any procided city.")
            wrong_matched_count += 1
        if publication.direction not in DIRECTIONS:
            logging.error(f"The direction {publication.direction} of the publication doesn't match any provided direction.")
            wrong_matched_count += 1
        if publication.date not in DATES:
            logging.error(f"The date {publication.date} of the publication doesn't match any provided date.")
            wrong_matched_count += 1
        if publication.temp < TEMP_LIMITS[0] or publication.temp > TEMP_LIMITS[1]:
            logging.error(f"The temp {publication.temp} of the publication doesn't fall within the imposed temp limits.")
            wrong_matched_count += 1
        if publication.rain < RAIN_LIMITS[0] or publication.rain > RAIN_LIMITS[1]:
            logging.error(f"The rain {publication.rain} of the publication doesn't fall within the imposed rain limits.")
            wrong_matched_count += 1
        if publication.wind < WIND_LIMITS[0] or publication.wind > WIND_LIMITS[1]:
            logging.error(f"The wind {publication.wind} of the publication doesn't fall within the imposed wind limits.")
            wrong_matched_count += 1
    
    print(f"The percentage of wrong publications generated is: {(wrong_matched_count / pub_count) * 100}%")
    print(f"The percentage of correct publications generated is: {((pub_count - wrong_matched_count) / pub_count) * 100}%")
    

def validate_thread_count():
    if SELECTED_THREAD_COUNT < MIN_THREAD_COUNT or SELECTED_THREAD_COUNT > MAX_THREAD_COUNT:
        logging.error("Selected number of threads is out of bounds!")
        exit(1)

def clean_file(file_path):
    with open(file_path, 'w') as file:
        pass

def write_file(file_path, data):
    with open(file_path, 'a') as file:
        file.write(data)

if __name__ == "__main__":
    validate_thread_count()
    clean_file("output.txt")
    publication_generator = PublicationGenerator(STATIONS, CITIES, DIRECTIONS, DATES, TEMP_LIMITS, WIND_LIMITS, RAIN_LIMITS)
    
    publication_count = int(config_object["PUBLICATIONSETUP"]["publication_count"])
    subscription_count = int(config_object["SUBSCRIPTIONSETUP"]["subscription_count"])

    publication_generator.generate_publications(publication_count)
    write_file("output.txt", f"PUBLICATIONS GENERATED: {publication_count}\n")
    [write_file("output.txt", f"{str(x)}\n") for x in publication_list]

    validate_generated_publications(publication_count, publication_list)

    subscription_generator = SubscriptionGeneratorV2(publication_generator, subscription_count, FIELD_WEIGHTS, OPERATOR_WEIGHTS)

    mapped_field_operator_weights = [t1 + t2 for t1, t2 in zip(subscription_generator.required_field_weights, subscription_generator.required_operator_weights)]
    mapped_field_operator_weights = sorted(mapped_field_operator_weights, key=lambda x: x[1], reverse=True)
    dd = subscription_generator.group_fields_by_weight(mapped_field_operator_weights)
    # for d in dd:
    #     print(d)
    # test = subscription_generator.generate_constraints()
    # for tt in test:
    #     for t in tt:
    #         print(str(t))

