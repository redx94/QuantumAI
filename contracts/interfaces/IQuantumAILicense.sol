// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IQuantumAILicense {
    function purchaseLicense(uint256 duration) external payable;
    function hasValidLicense(address user) external view returns (bool);
    function royaltyInfo(uint256 tokenId, uint256 salePrice) 
        external 
        view 
        returns (address receiver, uint256 royaltyAmount);
}
