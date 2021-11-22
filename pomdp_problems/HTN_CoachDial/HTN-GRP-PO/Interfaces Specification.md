"""
Hierarchical Task Recognition and Planning in Smart Homes with Partially Observability
Author: Dan Wang danwangkoala@gmail.com (May 2016 - June 2017)
Supervised by Prof. Jesse Hoey (https://cs.uwaterloo.ca/~jhoey/)
Association: Computer Science, University of Waterloo.
Research purposes only. Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by AGEWELL Networks of Centers of Excellence (NCE).
"""

Interfaces Specification, Used by The Tracking Algorithm
========================
By Dan Wang, Aug 25th, under updating


Notification Set
----------------
**Example**
	
    [{
            "ob_name":"faucet_1",
            "reliability":"0.9",
            "attribute": "state",
            "previous": "off",
            "current": "on"}]

**Format**

 - [{n1}, {n2}, {n3}...], it is a list, each element is a notification
 - Each notification {} include:
	 - "ob_name": **string**, indicate the name of a room object
	 - "reliability": **float**, indicate the attached sensor reliability
	 - "attribute": **string**, indicate the name of the attribute of the object that the sensor is monitor on
	 - "previous": **string**, indicate the previous sensor reading value
	 -  "current": **string**, indicate the current sensor reading value

**Notification Interface**

 - get_one_notif()
     - function: return the first notification set in the Queue (**do not delete it!**) if it has element
	 - input: None
	 - output: 
		 - content: A notification set / None
		 - type: list
 - delete_one_notif()
	 - function: **delete** the first notification set in the Queue if there is element in the Queue
	 - input: None
	 - output: None


Sensor Reading related
-----------------------

**"Sensor Reading" Example**
	
    {
	    "ob_name":"hand_1",
	    "attri_name":"soapy",
	    "value":["no", 2],
	    "reliability":0.9
	}

 **"Sensor Reading: Format**

 - An json item, includes:
	 - "ob_name": **string**, indicate the name of a room object
	 - "attri_name": **string**, indicate the interested attribute name
	 - "value":  **list**, with length 2
		 - element 1: **string**, the value of the sensor reading for this attribute
		 - element 2: **int**, the number of possible values of this attribute
	 - "reliability": **float**, indicate the reliability of this sensor

**Sensor reading interface**

 - get_sensor_reading(ob_name, attri_name)
	 - function: return the specific sensor reading
	 - input: "ob_name", "attri_name"
	 - output: an "**sensor reading**" item / None
