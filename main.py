import numpy as np
import operator
import logging
from configparser import ConfigParser
import threading
from collections import deque
import time

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
    "city": config_object["FIELDVALIDOPERATORS"]["opcity"].split(),
    "direction": config_object["FIELDVALIDOPERATORS"]["opdirection"].split(),
    "date": config_object["FIELDVALIDOPERATORS"]["opdate"].split(),
    "temp": config_object["FIELDVALIDOPERATORS"]["optemp"].split(),
    "wind": config_object["FIELDVALIDOPERATORS"]["opwind"].split(),
    "rain": config_object["FIELDVALIDOPERATORS"]["oprain"].split()
}

field_filter = ["fwstationIds", "fwcities", "fwdirections", "fwdates", "fwtemp", "fwwind", "fwrain"]
FIELD_WEIGHTS = []
for ff in field_filter:
    sum = 0
    try:
        tup = tuple(float(val) if "." in val else val for val in config_object["FIELDWEIGHTS"][ff].split())
        FIELD_WEIGHTS.append(tup)
        sum += tup[1]
    except KeyError:
        pass
    if sum < 1.0:
        logging.error("Not enough field weight.")
        exit(1)

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

PUBLICATION_COUNT = int(config_object["PUBLICATIONSETUP"]["publication_count"])
SUBSCRIPTION_COUNT = int(config_object["SUBSCRIPTIONSETUP"]["subscription_count"])

OUTPUT_FILE = config_object["OUTPUTFILE"]["output_file"]

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
        """Constructor for a Publication instance.

        Args:
            station_id (int, optional): Provided station_id . Defaults to None.
            city (str, optional): Provided city. Defaults to None.
            temp (int, optional): Provided temp. Defaults to None.
            rain (float, optional): Provided rain. Defaults to None.
            wind (int, optional): Provided wind. Defaults to None.
            direction (str, optional): Provided direction. Defaults to None.
            date (str, optional): Provided date. Defaults to None.
        """
        self.station_id = station_id
        self.city = city
        self.temp = temp
        self.rain = rain
        self.wind = wind
        self.direction = direction
        self.date = date
    
    def __str__(self) -> str:
        """String representation of a Publication instance.

        Returns:
            str: A string representation of a Publication instance.
        """
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
        """Constructor for a PublicationGenerator instance.

        Args:
            station_pool (list): A list of station ids.
            city_pool (list): A list of cities.
            direction_pool (list): A list of directions.
            date_pool (list): A list of dates.
            temp_limits (tuple): A tuple of temp limits, lower an upper bound.
            wind_limits (tuple): A tuple of wind limits, lower an upper bound.
            rain_limits (tuple): A tuple of rain limits, lower an upper bound.
        """
        self.station_pool = station_pool
        self.city_pool = city_pool
        self.direction_pool = direction_pool
        self.date_pool = date_pool
        self.temp_limits = temp_limits
        self.wind_limits = wind_limits
        self.rain_limits = rain_limits

    def generate_publication(self):
        """Generate a Publication instance with values available in the
        PublicationGenrator field pools.

        Returns:
            Publication: A new Publication instance.
        """
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
        """Generate number_of_publications publication instances
        and extend the current global list of publications:
        publication_list.

        Args:
            number_of_publications (int): Number of publications that need
            to be generated by the current thread.
        """
        publication_list.extend([self.generate_publication() for _ in range(number_of_publications)])

    def generate_publications(self, number):
        """Take the number given as a parameter and split it
        with the current amount of threads that need to be used for 
        generation. If a publication is lost due to rounding, add it
        to the first thread.
        Each thread available will use it to generate the
        specified number of publications it is given.

        Args:
            number (int): Total number of publications that need to be
            generated by the PublicationGenerator.

        Returns:
            list: A list of generated publications.
        """
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
        """Constructor for a Contraint.

        Args:
            factor (str): String field representation of a Publication field.
            operator (str): String field representation of an operator.
            required_value (int/str): Value that should consider the requirements
            of the factor and operator available for it.
        """
        self.factor = factor
        self.operator = operator
        self.required_value = required_value

    def __str__(self) -> str:
        """String representation of a Constraint.

        Returns:
            str: A strin representation of the Constraint instance.
        """
        return f"(factor: {self.factor}, operator: {self.operator}, value: {self.required_value});"

class Subscription:
    def __init__(self, constraints=None) -> None:
        """Constructor for a Subscription.

        Args:
            constraints (list, optional): A list of constraints for the subscription. Defaults to None.
        """
        self.constraints = constraints

    def __str__(self) -> str:
        """String representation of a Subscription that also displays
        it's constraints.

        Returns:
            str: A string representation of the Subscription that also
            displays it's constraints.
        """
        reprt = "Subscription Constraints: ("
        for constraint in self.constraints:
            reprt += str(constraint)
        return reprt + ")"
    
    def add_constraint(self, constraint):
        """Add a contraint to the current list of constraints.
        If the list of constraints is None, instantiate it.

        Args:
            constraint (Constraint): A Constraint instance.
        """
        if self.constraints == None:
            self.constraints = list()
        self.constraints.append(constraint)
    
    def is_constraint_respected(self, position, publication):
        """Evaluate a subscription constraint and see if the publication
        given respects it. The position represents the position in the constraints
        list.
        Since the operator and factor of the constraint are strings, it is not possible
        to directly evaluate the constraint.
        Each operator is mapped to it's operator operation using the operator_dict dictionay.
        The value of a Publication attribute can be found using getattr().
        And then the constraint can be evaluated by using the current_operator obtained
        with the value of the attribute and the required_value of the constraint.

        Args:
            position (int): The position in the constraint list.
            publication (Publication): A Publication instance.

        Returns:
            bool: True if the constraint is respected by the Publication,
            false if not.
        """
        current_constraint = self.constraints[position]
        current_operation = operator_dict.get(current_constraint.operator)
        evaluation_string = "current_operation(getattr(publication, current_constraint.factor), current_constraint.required_value)"
        return eval(evaluation_string)
    
class SubscriptionGeneratorV2:
    def __init__(self, publication_generator=None, subscription_count=None, required_field_weights=None, required_operator_weights=None) -> None:
        """Constructor for the SubscriptionGeneratorV2. This is the second version of the
        previous implementation.

        Args:
            publication_generator (PublicationGenrator, optional): A PublicationGenerator instance. Defaults to None.
            subscription_count (int, optional): Number of subscriptions that need to be generated. Defaults to None.
            required_field_weights (list, optional): List of mapped fields and weights. Defaults to None.
            required_operator_weights (list, optional): List of mapped operators and weights. Defaults to None.
        """
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
        """For each constraint provided in the group, a number
        of Constraints are generated based on the field_weight
        of the field.
        Constraints also respect the operator_weight provided.
        Since only an operator and weight is provided, it means
        that the rest of constraints would have any other operator
        available so that the field_weight is respected.

        Args:
            group (list): A list containting fields
            whose weights have a sum or less than 1.0.
        """
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
        """After the Constraints are generated, they need to be compressed
        or rather assigned to a correct Subscription without repeating.
        Firstly, the first subscriptions are generated from the first
        list of constraints available in the constraint_pool.
        After that the rest of Subscriptions are assigned from the next
        constraint lists until a the number of Subscriptions matched the
        subscription_count given.
        If the Subscription count is satisfied after the second batch
        and it still has elements, it is made sure that those
        constraints are not reasigned to existent Subscriptions
        that have such constraint (it is also helped by the fact that
        the constraint_pool list is sorted based on the number of constraints
        inside each sublist).
        At the end, the subscription_list is shuffled in  order to
        spread constraints in a randomly manner.

        Returns:
            list: A list of Subscriptions with correctly assigned
            constraints.
        """
        constraint_pool = self.generate_constraints()

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
    """Validates a publication list. Checks if all properties
    have correct values assigned from the ones available.

    Args:
        pub_count (int): Numbe rof publications that needed to be generated.
        publications (list): The list of Publications that need to be validated.
    """
    if len(publications) != pub_count:
        logging.warning(f"The number of requested publications: {pub_count} does not match the number of publications generated: {len(publications)}")
    
    wrong_matched_count = 0
    for publication in publications:
        wrong_flag = False
        if publication.station_id not in STATIONS:
            logging.error(f"The station id {publication.station_id} of the publication doesn't match any provided station id.")
            wrong_flag = True
        if publication.city not in CITIES:
            logging.error(f"The city {publication.city} of the publication doesn't match any procided city.")
            wrong_flag = True
        if publication.direction not in DIRECTIONS:
            logging.error(f"The direction {publication.direction} of the publication doesn't match any provided direction.")
            wrong_flag = True
        if publication.date not in DATES:
            logging.error(f"The date {publication.date} of the publication doesn't match any provided date.")
            wrong_flag = True
        if publication.temp < TEMP_LIMITS[0] or publication.temp > TEMP_LIMITS[1]:
            logging.error(f"The temp {publication.temp} of the publication doesn't fall within the imposed temp limits.")
            wrong_flag = True
        if publication.rain < RAIN_LIMITS[0] or publication.rain > RAIN_LIMITS[1]:
            logging.error(f"The rain {publication.rain} of the publication doesn't fall within the imposed rain limits.")
            wrong_flag = True
        if publication.wind < WIND_LIMITS[0] or publication.wind > WIND_LIMITS[1]:
            logging.error(f"The wind {publication.wind} of the publication doesn't fall within the imposed wind limits.")
            wrong_flag = True
        if wrong_flag:
            wrong_matched_count += 1
    
    print(f"The percentage of wrong publications generated is: {(wrong_matched_count / pub_count) * 100}%")
    print(f"The percentage of correct publications generated is: {((pub_count - wrong_matched_count) / pub_count) * 100}%")

def validate_generated_subscriptions(sub_count, subscriptions):
    """Validates a subsciption list. Checks if all properties
    have correct values assigned from the ones available.

    Args:
        sub_count (int): Number of Subscriptinos that needed to be generated.
        subscriptions (list): The list of Subscriptinos that needed to be validated.
    """
    if sub_count != len(subscriptions):
        logging.warning(f"The number of requested publications: {sub_count} does not match the number of publications generated: {len(subscriptions)}")

    wrong_matched_count = 0
    for subscription in subscriptions:
        wrong_flag = False
        for constraint in subscription.constraints:
            if constraint.factor not in VALID_OPERATIONS:
                logging.error(f"The constraint factor '{constraint.factor}' of the subscription doesn't match any available constraint factors.")
                wrong_flag = True
                continue
            if constraint.operator not in VALID_OPERATIONS[constraint.factor]:
                print(constraint.operator)
                print(VALID_OPERATIONS[constraint.factor])
                logging.error(f"The constraint operator '{constraint.operator}' of the subscription doesn't match any available constraint operators.")
                wrong_flag = True
            match constraint.factor:
                case "city":
                    if constraint.required_value not in CITIES:
                        logging.error(f"The constraint value '{constraint.required_value}' of the subscription doesn't match any provided city value.")
                        wrong_flag = True
                case "stationId":
                    if constraint.required_value not in STATIONS:
                        logging.error(f"The constraint value '{constraint.required_value}' of the subscription doesn't match any provided station_id value.")
                        wrong_flag = True
                case "direction":
                    if constraint.required_value not in DIRECTIONS:
                        logging.error(f"The constraint value '{constraint.required_value}' of the subscription doesn't match any provided direction value.")
                        wrong_flag = True
                case "date":
                    if constraint.required_value not in DATES:
                        logging.error(f"The constraint value '{constraint.required_value}' of the subscription doesn't match any provided date value.")
                        wrong_flag = True
                case "temp":
                    if constraint.required_value < TEMP_LIMITS[0] or constraint.required_value > TEMP_LIMITS[1]:
                        logging.error(f"The constraint value '{constraint.required_value}' of the subscription doesn't fall within the imposed temp limits.")
                        wrong_flag = True
                case "wind":
                    if constraint.required_value < WIND_LIMITS[0] or constraint.required_value > WIND_LIMITS[1]:
                        logging.error(f"The constraint value '{constraint.required_value}' of the subscription doesn't fall within the imposed wind limits.")
                        wrong_flag = True
                case "rain":
                    if constraint.required_value < RAIN_LIMITS[0] or constraint.required_value > RAIN_LIMITS[1]:
                        logging.error(f"The constraint value '{constraint.required_value}' of the subscription doesn't fall within the imposed rain limits.")
                        wrong_flag = True
                case _:
                    logging.error(f"The constraint value '{constraint.required_value}' of the subscription doesn't match any available constraint values or limits.")
                    wrong_flag = True

        if wrong_flag:
            wrong_matched_count += 1
    
    print(f"The percentage of wrong subscriptions generated is: {(wrong_matched_count / sub_count) * 100}%")
    print(f"The percentage of correct subscriptions generated is: {((sub_count - wrong_matched_count) / sub_count) * 100}%")

def validate_thread_count():
    """Validates the number of threads given in the config. If it
    is a value that doesn't match the requirements, the program stops.
    """
    if SELECTED_THREAD_COUNT < MIN_THREAD_COUNT or SELECTED_THREAD_COUNT > MAX_THREAD_COUNT:
        logging.error("Selected number of threads is out of bounds!")
        exit(1)

def validate_generation_counts():
    """Validates the number of subscriptions and publications to be 
    generated that were given in the config. If it is a value that 
    is zero, the program stops.
    """
    publication_count = int(config_object["PUBLICATIONSETUP"]["publication_count"])
    subscription_count = int(config_object["SUBSCRIPTIONSETUP"]["subscription_count"])
    if publication_count <= 0 or subscription_count <= 0:
        logging.error("Selected number of subscriptions/publications to be generated is not a valid value.")
        exit(1)

def clean_file(file_path):
    """Empty a file before writting content.
    Also serves as a file creation method.

    Args:
        file_path (str): Name of the file/path to the file.
    """
    with open(file_path, 'w') as file:
        pass

def write_file(file_path, data):
    """Append data to a file.

    Args:
        file_path (str): Name of the file/path to the file.
        data (str): Data that needs to be written in the file.
    """
    with open(file_path, 'a') as file:
        file.write(data)

if __name__ == "__main__":
    validate_thread_count()
    validate_generation_counts()

    clean_file(OUTPUT_FILE)

    publication_generator = PublicationGenerator(STATIONS, CITIES, DIRECTIONS, DATES, TEMP_LIMITS, WIND_LIMITS, RAIN_LIMITS)

    start = time.time()
    publication_generator.generate_publications(PUBLICATION_COUNT)
    end = time.time()
    write_file(OUTPUT_FILE, f"PUBLICATIONS GENERATED: {len(publication_list)}\n")
    write_file(OUTPUT_FILE, f"Time spent generating: {end - start}ms\n")
    [write_file(OUTPUT_FILE, f"{str(x)}\n") for x in publication_list]

    validate_generated_publications(PUBLICATION_COUNT, publication_list)

    subscription_generator = SubscriptionGeneratorV2(publication_generator, SUBSCRIPTION_COUNT, FIELD_WEIGHTS, OPERATOR_WEIGHTS)

    start = time.time()
    subscription_list = subscription_generator.compress_generated_constraints()
    end = time.time()
    write_file(OUTPUT_FILE, f"SUBSCRIPTIONS GENERATED: {len(subscription_list)}\n")
    write_file(OUTPUT_FILE, f"Time spent generating: {end - start}ms\n")
    [write_file(OUTPUT_FILE, f"{str(x)}\n") for x in subscription_list]
    validate_generated_subscriptions(SUBSCRIPTION_COUNT, subscription_list)

    

