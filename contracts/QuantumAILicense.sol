// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/interfaces/IERC2981.sol";

contract QuantumAILicense is ERC1155, Ownable, IERC2981 {
    uint256 public constant LICENSE_ID = 1;
    uint256 public licenseFee;
    uint256 public royaltyPercentage;
    uint256 public constant GRACE_PERIOD = 3 days;
    
    struct LicenseDetails {
        uint256 expiry;
        uint256 usageLimit;
        uint256 usageCount;
        bool isSubscription;
    }
    
    mapping(address => LicenseDetails) public licenses;
    mapping(address => bool) public blacklisted;
    
    event LicensePurchased(address indexed buyer, uint256 expiry);
    event LicenseRevoked(address indexed holder);
    event RoyaltyPaid(address indexed recipient, uint256 amount);
    event UsageRecorded(address indexed user, uint256 count);

    constructor(uint256 _licenseFee, uint256 _royaltyPercentage) 
        ERC1155("https://api.quantumai.com/license/metadata/{id}.json") 
    {
        licenseFee = _licenseFee;
        royaltyPercentage = _royaltyPercentage;
    }

    function purchaseLicense(uint256 duration, bool subscription) external payable {
        require(msg.value >= calculateFee(duration, subscription), "Insufficient payment");
        require(duration >= 30 days, "Minimum 30 days");
        require(!blacklisted[msg.sender], "Address blacklisted");

        uint256 expiry = block.timestamp + duration;
        if (balanceOf(msg.sender, LICENSE_ID) == 0) {
            _mint(msg.sender, LICENSE_ID, 1, "");
        }
        
        licenses[msg.sender] = LicenseDetails({
            expiry: expiry,
            usageLimit: subscription ? type(uint256).max : 1000,
            usageCount: 0,
            isSubscription: subscription
        });
        
        emit LicensePurchased(msg.sender, expiry);
    }

    function renewLicense(uint256 duration) external payable {
        require(balanceOf(msg.sender, LICENSE_ID) > 0, "No license to renew");
        require(msg.value >= licenseFee * duration / 30 days, "Insufficient payment");
        require(!blacklisted[msg.sender], "Address blacklisted");

        uint256 newExpiry = block.timestamp + duration;
        if (licenses[msg.sender].expiry > block.timestamp) {
            newExpiry = licenses[msg.sender].expiry + duration;
        }
        licenses[msg.sender].expiry = newExpiry;
        
        emit LicensePurchased(msg.sender, newExpiry);
    }

    function revokeLicense(address user) external onlyOwner {
        require(balanceOf(user, LICENSE_ID) > 0, "No license found");
        _burn(user, LICENSE_ID, 1);
        delete licenses[user];
        emit LicenseRevoked(user);
    }

    function recordUsage(address user, uint256 count) external onlyOwner {
        require(hasValidLicense(user), "Invalid license");
        licenses[user].usageCount += count;
        emit UsageRecorded(user, count);
    }

    function hasValidLicense(address user) public view returns (bool) {
        LicenseDetails memory details = licenses[user];
        return balanceOf(user, LICENSE_ID) > 0 && 
               !blacklisted[user] &&
               details.expiry + GRACE_PERIOD > block.timestamp &&
               details.usageCount < details.usageLimit;
    }

    function royaltyInfo(uint256, uint256 salePrice) 
        external 
        view 
        override 
        returns (address receiver, uint256 royaltyAmount) 
    {
        return (owner(), (salePrice * royaltyPercentage) / 100);
    }

    function withdrawFunds() external onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }

    function blacklistAddress(address user) external onlyOwner {
        blacklisted[user] = true;
        if (balanceOf(user, LICENSE_ID) > 0) {
            _burn(user, LICENSE_ID, 1);
        }
        emit LicenseRevoked(user);
    }

    function calculateFee(uint256 duration, bool subscription) public view returns (uint256) {
        if (subscription) {
            return licenseFee * duration / 30 days;
        } else {
            return licenseFee * duration / 30 days / 2;
        }
    }
}
