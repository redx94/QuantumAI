import { expect } from "chai";
import { ethers } from "hardhat";
import { QuantumAILicense } from "../typechain-types";

describe("QuantumAILicense", function () {
  let license: QuantumAILicense;
  const licenseFee = ethers.utils.parseEther("5");
  const royaltyPercentage = 20;

  beforeEach(async function () {
    const License = await ethers.getContractFactory("QuantumAILicense");
    license = await License.deploy(licenseFee, royaltyPercentage);
    await license.deployed();
  });

  it("Should allow license purchase", async function () {
    const [, buyer] = await ethers.getSigners();
    const duration = 30 * 24 * 60 * 60; // 30 days

    await expect(license.connect(buyer).purchaseLicense(duration, {
      value: licenseFee
    })).to.emit(license, "LicensePurchased");

    expect(await license.hasValidLicense(buyer.address)).to.be.true;
  });
});
