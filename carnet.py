from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("KeyPresent", "Starts"),
        ("Starts","Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)
cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        [0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.001, 0.001],
        [0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.999, 0.999]
    ],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={"Starts": ['yes', 'no'], "Ignition": ["Works", "Doesn't work"], "Gas": ['Full', "Empty"], "KeyPresent": ['yes', 'no']}
)


cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)

cpd_key_present = TabularCPD(
    variable="KeyPresent",
    variable_card=2,
    values=[[0.70], [0.30]],
    state_names={"KeyPresent": ["yes", "no"]}
)



# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_key_present)

car_infer = VariableElimination(car_model)


if __name__ == "__main__":

    print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

    # Add 5 Queries
    #1
    q = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
    print("Probability that the battery is not working given that the car will not move:")
    print(q)

    #2
    q1 = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
    print(q1)

    #3
    q3 = car_infer.query(variables=["Radio"], evidence={"Gas": "Full", "Battery": "Works"})
    print(q3)

    #4
    q4 = car_infer.query(variables=["Ignition"], evidence={"Gas": "Empty","Moves": "no"})
    print(q4)

    #5
    q5 = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
    print(q5)

    # Key Present Query
    q6 = car_infer.query(variables=["KeyPresent"], evidence={"Moves":"no"})
    print("Probability that the key is not present given that the car does not move:")
    print(q6)





