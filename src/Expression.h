#ifndef _SURFACE3D_EXPRESSION_H_
#define _SURFACE3D_EXPRESSION_H_
#include <map>
#include <vector>

/*!
* \var TokenType 
* It's used to represent Token types in the Expression::type array
*/
enum TokenType
{
	NUMBER = 1,
	VARIABLE = 2,
	OPERATOR = 3,
	FUNCTION = 4,
	BRACKET = 5
};

class Expression {
private:
	/*!
	* \brief Checks if a given char is a math operator
	* \param input A character
	* \retval true The given character is + - * / ^ or %
	* \retval false For any other character
	*/
	bool is_operator(const char * input);
	/*!
	* \brief Gets the next alphabetical substring from the given string and checks if it exists in the function table
	* \param[in] input A pointer to a char array to check
	* \retval true  If the first alphabetical substring of the input exists in the function table
	* \return false If it doesn't exist
	*/
	bool is_function(const char * input);
	/*!
	* \brief Checks if the next alphabetical substring is the name of a constant
	* \param[in] input A pointer to char array
	* \retval true If the first alphabetical substring of the input exists in the constants table
	* \retval false If it doesn't exist
	*/
	bool is_const(const char * input);
	/*!
	* \brief Checks if the next character is a variable
	* \param[in] input
	* \retval true If the next character is x,y,z,t 
	* \retval false If it isn't 
	*/
	bool is_variable(const char * input);
	/*!
	* \brief Checks if the next character is precedence operator
	* \param[in] input A pointer to input char array
	* \retval true If the next char is ( or ) 
	* \retval false If it isn't
	*/
	bool is_bracket(const char * input);
	/*!
	* \brief Gets the next alphabetical substring of a given char array
	* \param[in] input The input char array
	* \return A std::string containing the next alphabetical substring
	*/
	std::string get_next_string(const char *& input);

	/*!
	* \var Expression::operators, Expression::functions, Expression::constants, Expression::variables
	* These 4 hash tables are used to encode tokens into numbers and decode them back 
	* This is neccessary as you can't pass function pointers from the _host_ to the _device_ 
	*/
	std::map<char, int> operators;
	std::map<std::string, int> functions;
	std::map<std::string, float> constants; 
	std::map<char, int> variables;
public:
	/*!
	* \var Expression::output 
	* A vector containing numerical representation of the tokens parsed from the input string
	*/
	std::vector<int> output;
	/*!
	* \var Expression::numbers
	* A vector containing all the constant numbers in the expression
	*/
	std::vector<float> numbers;
	/*!
	* \var Expression::type
	* A vector containing numbers corresponding to the elements of the output vector 
	* It indicated the type of the token
	* The possible values are specified in the TokenType enum
	*/
	std::vector<int> type;
	/*!
	* \brief Expression constructor. Parses a math expression from string 
	*/
	Expression(const char * input);
	/*!
	* \brief Expression destructor 
	*/
	~Expression();
};


#endif