pragma solidity ^0.8.17;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract QAIToken is ERC20 {
    constructor(uint256 initialSupply) ERC20("Quantum AI Token", "QAI") {
        _mint(msg.sender, initialSupply);
    }
}