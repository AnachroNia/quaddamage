/*!
* \file
* A simple math expression parser. For a detailed description of the class and
* its methods, please check the Expression.h file
*/
#include "Expression.h"
#include <cstdlib>
#include <cctype>
#include <iostream>

bool Expression::is_operator(const char * input){
	return *input == '+' || *input == '-' || *input == '*' || *input == '/' || *input == '^' || *input == '%';
}

bool Expression::is_function(const char * input){
	const char * temp = input;
	std::string s = get_next_string(input);
	input = temp;
	if (functions.count(s) != 0) return true;
	else return false; 
}

bool Expression::is_const(const char * input){
	const char * temp = input;
	std::string s = get_next_string(input);
	input = temp;
	if (constants.count(s) != 0) return true;
	else return false; 
}

bool Expression::is_variable(const char * input){
	if (*input == 'x' || *input == 'y' || *input == 'z' || *input == 't') return true;
	return false;
}

bool Expression::is_bracket(const char * input){
	if (*input == '(' || *input == ')') return true; 
	return false; 
}

std::string Expression::get_next_string(const char *& input){
	std::string s;
	while (isalpha(*input)){
		s += input[0];
		input++;
	}
	return s;
}

Expression::Expression(const char * input){
	// Using Tables to pass functions from _host_ to _device_ 
	// Note that the values assigned to the operators are used to determine the order of evaluation too
	operators['+'] = 0;
	operators['-'] = 1;
	operators['/'] = 2;
	operators['*'] = 3;
	operators['%'] = 4; //Modulo 
	operators['^'] = 5; //Power 
	//A more frequently used functions should be higher in the list 
	//The Evaluate kernel uses endless if/else instead of function pointers 
	//as they appear to be slower in cuda kernels
	functions["sin"] = 1;
	functions["cos"] = 2;
	functions["tan"] = 3;
	functions["ctan"] = 4;
	functions["arcsin"] = 5;
	functions["arccos"] = 6;
	functions["arctan"] = 7;
	functions["arcctan"] = 8;
	functions["sqrt"] = 9;

	variables['x'] = 0;
	variables['y'] = 1;
	variables['z'] = 2;
	variables['t'] = 3; //Time - Reserved for Function Animation

	while (*input){
		if (isdigit(*input)){
			char * it; 
			double number = strtod(input, &it);
			numbers.push_back(number);
			output.push_back(numbers.size() - 1); 
			type.push_back(TokenType::NUMBER);
			input = it;
		}
		else if(is_operator(input)) {
			output.push_back(operators[*input]);
			type.push_back(TokenType::OPERATOR);
			input++;
		}
		else if (is_function(input)) {
			output.push_back(functions[get_next_string(input)]);
			type.push_back(TokenType::FUNCTION);
		}
		else if (is_const(input)){
			output.push_back(constants[get_next_string(input)]); //Replace with the proper number 
			type.push_back(TokenType::NUMBER);
		}
		else if (is_variable(input)){
			output.push_back(variables[*input]);
			type.push_back(TokenType::VARIABLE);
			input++;
		}
		else if (is_bracket(input)){
			if (*input == '(') output.push_back(0); //0 = openning bracket
			else output.push_back(1);               //1 = closing bracket 
			type.push_back(TokenType::BRACKET);
			input++;
		}
		else {
			std::cout << "Cannot parse string: " << *input << std::endl; 
			break;
		}
	}
}

Expression::~Expression(){}