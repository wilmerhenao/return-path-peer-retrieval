# Baseline-anchored multimodal heads: historical run report

This report is generated from the machine-readable run artifact. It is a
historical diagnostic and does not authorize a modern evaluation or paper edit.

## Run status

| Field | Artifact value |
|---|---|
| run_id | run_20260802_v2 |
| status | STOP |
| started_at_utc | 2026-08-03T03:27:16.752169+00:00 |
| completed_at_utc | 2026-08-03T03:37:00.433111+00:00 |

## Provenance

| Field | Artifact value |
|---|---|
| protocol_tag | multimodal-peer-protocol-v2-pre-results + multimodal-peer-protocol-v2-clarification-1 |
| peeled_commit | redacted-for-anonymous-release |
| git_status_short | ?? experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v2/ |
| configuration_sha256 | d094b78050ecfd25760dc1304b7a5214649a13928dfb0cc8aaa14a6e4d76c0f8 |
| manifest_sha256 | df2e974f4e9e4503f23c1265e414f41f42cdad8f5f0cba951a7cd384494cf3f3 |

### Source identities

| Field | Artifact value |
|---|---|
| .gitignore | 3dfcaa47566324b7b71c36548ece56f623a4cbed7e14964a19dcbacc0d954276 |
| environment/core.lock | f2821f06e34eed9b1dbe48ba2b1fef891015cc63b6957ee9154974c4381548b3 |
| environment/models.lock | aee8565fb9a2c72244af98b1e5a80b09335d7b0fc266941270f97c4b61da16ab |
| experiments/multimodal_peer_encoder/.gitignore | 2cba777134b237dbd0463f838cd043c1d8ce9cc254947a603f5ae66c3b44baef |
| experiments/multimodal_peer_encoder/PROTOCOL_v2.md | 73e9bbd7abe6ddb63ff4c8c4a15981a921ed7de608d650b5f23f4c65fcfbe3ba |
| experiments/multimodal_peer_encoder/PROTOCOL_v2_CLARIFICATION_1.md | 33e4b83d5c92d2a103cdbf1a818a20c34af1d4b56338368daa9d0ea7c78ec9b2 |
| experiments/multimodal_peer_encoder/README.md | 029d17819c5766c8e9ad17a5c188cbeeffb0721dd791b054c6bd8c0e08522ec3 |
| experiments/multimodal_peer_encoder/__init__.py | e29d32daa06df5be47ecc559381f2d12faa47874ce5cba02f5a9353bf8495551 |
| experiments/multimodal_peer_encoder/build_historical_embeddings.py | 5452f98055732a5f02244854072adb1efb3a5c32df9ec36046cf484de88de1f1 |
| experiments/multimodal_peer_encoder/config/v2_historical_protocol.json | d094b78050ecfd25760dc1304b7a5214649a13928dfb0cc8aaa14a6e4d76c0f8 |
| experiments/multimodal_peer_encoder/evaluation.py | 257715b99c1a36ba803312d20ea34323726cde76321231bf9dc86ba2bebe68fe |
| experiments/multimodal_peer_encoder/historical_data.py | 35962dfa3e3fd3d29a1fe544b041e3eda6658236aefed124397709b20fce69e1 |
| experiments/multimodal_peer_encoder/model.py | 8714f0fdf9616c89f4684bc1f249c43eaa8d63700e79585fd98ad83208ad2b0a |
| experiments/multimodal_peer_encoder/report_v2.py | 02e2a9c6c1e44fae5a1317810a01ab6f409c8f53cd65fed3c899a2bdf3c5b732 |
| experiments/multimodal_peer_encoder/requirements.lock | c6f339606aeb5b78607c3f6adc9b429794df35fc69506eb5a1700ca530492000 |
| experiments/multimodal_peer_encoder/run_v2_historical.py | 54ec63e989ade42851c84c7806f9904982085d1780ae460087a68bc1d4eff564 |
| experiments/multimodal_peer_encoder/tests/test_v2_data.py | 8ee3260e681434e244fac84a222a67a65da26d8478901d939d6ce6aa0bdf3151 |
| experiments/multimodal_peer_encoder/tests/test_v2_model.py | 2a91f627028902f694b11201b34d2c7fc4b0aeb10efb3a7d7ca107a3ef11364a |
| experiments/multimodal_peer_encoder/tests/test_v2_report.py | d6bcc6db997df28dd4691bdc22b3ffdd2dc615d95d0bde3c3db31c2a99286c3d |
| experiments/multimodal_peer_encoder/tests/test_v2_runner.py | 55f62fc8db487781b3d6a6da740c68d4388c561ecc2fa48c563a89d984cc47f0 |
| experiments/multimodal_peer_encoder/tests/test_v2_training.py | 228a2850edd065629cac9e15cb75e4ad1f94835aa74048edf9faf4fa262f826b |
| experiments/multimodal_peer_encoder/v2_data.py | d558a24f76754ca6bba3ea828aec9953902531ab9e2507358f3e221489d8bf68 |
| experiments/multimodal_peer_encoder/v2_diagnostics.py | 0a37794a6901c9421aa37a2cf7280878e2d638664e05ddd4d7a98678b2c5e5a0 |
| experiments/multimodal_peer_encoder/v2_model.py | 938c37b584b223b08e6ac6a37a1f49ee0e8c81ace0ca1b59e9d846f657ba10ef |
| experiments/multimodal_peer_encoder/v2_self_tests.py | 64dc5e55bd882914dd9441e751b2e29f3d21071652870828d2c71189df48c7b6 |
| experiments/multimodal_peer_encoder/v2_training.py | 40118553c7ad6c17ad02e72f1f450bd1d7a882ffada85c1184f10fd050cb633a |
| scripts/__init__.py | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| scripts/build_anonymous_release.py | 6c23c3180a50547f7df7d24752b2563bca4fe3919045b72ce81252194bc02e35 |
| scripts/metrics.py | 827e411d77466675d5837285f3c73a83d7a28fdd4d9739076dc8fd8b384b62c6 |
| tests/test_release_builder.py | da89029567ac61e51e7d564c74c69d9a512b4afa2a0ebb60eee3ee6f327ca2be |

### Input identities

| Field | Artifact value |
|---|---|
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H10_eligible_tickers.json | e0ee425d0b6db20b210aae9da27a5f8610485ccf52984e40066d3dcc56b2b7fa |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H10_granite_512.npy | 71743b30a806967bb7bba6b1a5963add923f458a82ef18d0a96afb0ae4174b97 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H11_eligible_tickers.json | 22f5d2fdd7121e524a512a11c39fe508f5c19ec8a97ce1a2c4e1aeb20cadc0d8 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H11_granite_512.npy | f1ffe0ec29b9ef3a9d12567cd0dffe7f0818152dcacc21b910f77286dbc02562 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H12_eligible_tickers.json | d76fdb4eb9d1928fb5bda319b045db6e98bffc308ba51a1e02d5c03690ccf030 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H12_granite_512.npy | f3934e50405e0ce98e946b57cc3c5dd6fe4eeba7ce36575766cc2fb52fc13480 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H1_eligible_tickers.json | 9f3811dbecbeb9a0efb656132492ef33508781be8394f832591a5ee3a5446384 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H1_granite_512.npy | 8f87663a610df186fd40b8f26847eb76cbda9eb76f52261c98db7061e1f5d3fc |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H2_eligible_tickers.json | e8bbac1b57b7eee41dd1392b3353e61ebe901eba3740186b10204c6be4c55f52 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H2_granite_512.npy | d2feaedb66a4eee1acf313a5d61e09274d0df76e622d9167a70aba52f1b42219 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H3_eligible_tickers.json | 1cb50f954842606c4162c65702ff648e971412bcbeaf7b3276aeaa9934cdd9d3 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H3_granite_512.npy | 8fac0a119070f772c960cb4d42d21dc1431b8b00f44b96b89fe6343c7801f727 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H4_eligible_tickers.json | 3449a8af81bfb28c9bbdf5599659bdb44ca0a0e0a4549b3db2720d1c3bdc0c16 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H4_granite_512.npy | 2e043fcf8dd97cc4c8ceedda6097c0a2c1a2618cd5e1fd3908b04b1dcd0b9753 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H5_eligible_tickers.json | aaca2301d53c36e594ec04a2ed03e8a6199ca5ecbcd3bfad699fe8647e986d46 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H5_granite_512.npy | beb7561ab13337ed6d276d29e1942ee7ba357411ea0c5018428a48c7e547ccaa |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H6_eligible_tickers.json | 415e672fd49ccf4bb381cd341da1d5d1362a34f9b00a38fc86f7f7afe79ac30e |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H6_granite_512.npy | dcc74a8e8f5e9f1fc908721a0b126bca7d73035c392d336c2374b589a02e2138 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H7_eligible_tickers.json | 4ba595b0fff5e367043e14fa82c095be5111e1e7523ed591571dd9d970b3a158 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H7_granite_512.npy | 876efd3fa4e0b950e65946770abe0fc8dd97f917b5fa0e6013f4978a60185e21 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H8_eligible_tickers.json | 20cfceca98c19ab45a82a997ea8be3fa2aa6b323f913c2261d1890151fabf8c1 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H8_granite_512.npy | 80486fba35fc25560619189b7393b3d23a49a8fdb688bae97f407096fb3b1952 |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H9_eligible_tickers.json | 7ad9230cfb0b4a42e53594c1f3fcef4b9fcfb8463a28223aeec8cf0ff568b13e |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/H9_granite_512.npy | c9748d407afbf727dc8db59868e6e6d09e276985f31f65acb3bb16264859736f |
| experiments/multimodal_peer_encoder/artifacts/historical_text_cache/cache_manifest.json | 0e32965b0d33f59b25243733b4dd687a29fcb4497dba8eb86e931f9fae131fd0 |
| experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v2/development_selection.json | 228e2a9a62e6cc523318ad54454a1cb7ad24ebd4e1974344c87e878ab84d2026 |
| results/ea_historical/windows/H1/meta.json | 5d184e26ebe94c6030b4555cc037af4531a5a64cecb89dbf541224d1343cbfea |
| results/ea_historical/windows/H1/ts_features.npy | 45c6aea1ea14dd4e809ba796e79af89ca14ca0605849ab5965fe574232aa3723 |
| results/ea_historical/windows/H1/universe.parquet | e12ebf956c4cc4b13075cd1a2e9e20c4cec59cf5c0d361f1ae77d93b96e9e563 |
| results/ea_historical/windows/H10/corr_target.npy | bd47dd6054539e6e1622500cb52b7a091a72954e985311ab05fb79aaa9783908 |
| results/ea_historical/windows/H10/meta.json | a6f43b2b7b3b8ed06f0a5fe4892a9854719a259c704ddb58186fc578c068a752 |
| results/ea_historical/windows/H10/ts_features.npy | 3eba7afd71790426024a1f5f1d2008fb3f7d9bedd7f4577cfbe0b91567340354 |
| results/ea_historical/windows/H10/universe.parquet | 3024bebcc708ca15825aa68419642508234aae67fcc32ece70e06f6d15257527 |
| results/ea_historical/windows/H11/corr_target.npy | 73ebd16599f01549522c0b94d8f7cd06e8252327bf9f43a752d669962e1bd94b |
| results/ea_historical/windows/H11/meta.json | 4fd6e2a8ad39f9924badfd2d00914b0b6ac3ca5a42e54a5d667068fefcfbd232 |
| results/ea_historical/windows/H11/ts_features.npy | 6b0c073a0f8cdbc7fe54b6015c5eee091c5e342e15b8edf30cec5867447de06c |
| results/ea_historical/windows/H11/universe.parquet | 7b271125977cda5e5653f86fd9c48089938bb52dc8004643ad0e567d0f31855f |
| results/ea_historical/windows/H12/corr_target.npy | 43aaf98b5a030d1fea6fad3515312f5252229f910ed603566b209775d8dc4204 |
| results/ea_historical/windows/H12/meta.json | 87e21ee064acfc0bc13abfc216a410b21605e10536853003363b8e0cbd517af7 |
| results/ea_historical/windows/H12/ts_features.npy | ed5094a04a83afc9bff2a73e868e41e445902396c03e5dbfd47b29a5eb7d604a |
| results/ea_historical/windows/H12/universe.parquet | c25d09c30cb453384ab45a62c5271e5dfe5d07712610464b5ae530c9dba96b4f |
| results/ea_historical/windows/H13/corr_target.npy | 99eb403116dbab8ab2105923880bf4036aec08e1eefd8f4b86aa7c9b57a9cd9a |
| results/ea_historical/windows/H13/meta.json | 43b74c390d75545aec7fd77864e2782df93e370e046dd27641793a128b117c18 |
| results/ea_historical/windows/H13/ts_features.npy | 3539db9d44c2b3e3adb20ff577280a7e0f8d70555b8e3c6e5c12111c30c92b31 |
| results/ea_historical/windows/H13/universe.parquet | cab4c41e6417d5442fc614b3d542c85fa915eb906af9de2c08e496efa0b1b844 |
| results/ea_historical/windows/H2/corr_target.npy | 5be2d396856bc850a5bce3c8d40752d532610f4bf0c0ccdd84b99861c4747f5f |
| results/ea_historical/windows/H2/meta.json | be17bed015b7dda4a26941d921a1770dcf1eae4be0a7bbf4e3d047aa1062609b |
| results/ea_historical/windows/H2/ts_features.npy | e7be1758e3f5f3a80b6c9a7b6eeab4aa457f54abe426017458fa97c77c3b8519 |
| results/ea_historical/windows/H2/universe.parquet | 0cf0439d9a95b414d15b2daf0b19b55725a1ef960edc33cd38abd370ee172256 |
| results/ea_historical/windows/H3/corr_target.npy | 355c56556270c35141417bbf2207c1af50237dd8db3d60befe5a0b1280efa5b6 |
| results/ea_historical/windows/H3/meta.json | 1fce8256cba239c75df6882b37d26e3bd5fd3dc38ce62738a3ecc21811760ba3 |
| results/ea_historical/windows/H3/ts_features.npy | f8b2f416915619871bc1da36bce5db81fbd124aa9092e643123dae2ce3021c62 |
| results/ea_historical/windows/H3/universe.parquet | 70d19f03b5875c56c8bf4598a78e8226e6dd026c97b245cbba6dc78677dbc9a3 |
| results/ea_historical/windows/H4/corr_target.npy | 81d1dc6803bfd6ec739b89e5b2edc34248ba620d942a850c3989ab39d954561c |
| results/ea_historical/windows/H4/meta.json | 62355406067b9b412c32b513bd18cbe37ae107e167554323ce86cc2bf6d97f09 |
| results/ea_historical/windows/H4/ts_features.npy | 2f11532431e18794cb3d3da2628b547ee7cb019b0ed59c4a636527aa1d7b1b1e |
| results/ea_historical/windows/H4/universe.parquet | de0fa1ab4aa8c38e98d1b1caa0d41dd34f01d3d82d08aee30f771a5708959166 |
| results/ea_historical/windows/H5/corr_target.npy | 4f8b24203731da427ce4c8252a56c1a832af7c6cc55ad9276f67a666a9698301 |
| results/ea_historical/windows/H5/meta.json | c0f76b102f7772b4b9e8b5d10c0ddd6bae291c68434536aaeeb9dc5f19778616 |
| results/ea_historical/windows/H5/ts_features.npy | 7a33be3ece39ad1898e5813d7fc1864a1f176532f41b760c1191e1d95ffab7f5 |
| results/ea_historical/windows/H5/universe.parquet | 73c9f6a6982183e386509e0177ca559a9134c4d75373b6e1ceca28f90bb4f81a |
| results/ea_historical/windows/H6/corr_target.npy | b0002d8e25a4f88ba5571ba76740aab3e252b129c9883d10684db79d3991bfe7 |
| results/ea_historical/windows/H6/meta.json | 399fd7d6b78ea66d6afc65745cc93990817e162ff2d5504e1a70b77877e87cc9 |
| results/ea_historical/windows/H6/ts_features.npy | f312aadf1bcf8af43dc7d3d9c0b210cee1e2f317b52823816356eb76c7158758 |
| results/ea_historical/windows/H6/universe.parquet | 2e603ca73c2b89ed316eb97a06be512462c26a38312511c045e052bad80e64fe |
| results/ea_historical/windows/H7/corr_target.npy | af40fe893e32b79cbde3197f2e260d402a79edd23067a41e9eb8735d252f9ac7 |
| results/ea_historical/windows/H7/meta.json | a305aa6412c52e9369cd05d0988a3d1e3855af16b2e46c2e08f1e8002783832f |
| results/ea_historical/windows/H7/ts_features.npy | 4ee47877593cc17bbaec78309755ba0993446651ffb9478db4c559343dce144a |
| results/ea_historical/windows/H7/universe.parquet | 900eb51f4c78b9c45b6cea43b0a595f0b6f3a883c4ce35a0b1452a92e013e7ff |
| results/ea_historical/windows/H8/corr_target.npy | f8e32579757eb1d0b08eec46b7553ba8f618b52516c05ca5174b421ffe5ac5aa |
| results/ea_historical/windows/H8/meta.json | 7fb96ea2aaff9111e8e2609b7bf595eee3f4c3b9475796e73ea79cc452421d19 |
| results/ea_historical/windows/H8/ts_features.npy | e8e51fe37eefad67807053435d24a2e7a69574d979f9818113ebd614b28a47c3 |
| results/ea_historical/windows/H8/universe.parquet | 338a21419e568c05b276d4ca8a01bb9482ea37d2a3ff41511cf8617b2816a846 |
| results/ea_historical/windows/H9/corr_target.npy | 82e9a0e46ab0da30505c41acaa70c8209afb677496692d1cc3baf77bf2fcb5a2 |
| results/ea_historical/windows/H9/meta.json | 64b2f4ce3a5324a756cd7f7a32eeff7511bca818f1ad1e3c84cc19c0bd3d3b03 |
| results/ea_historical/windows/H9/ts_features.npy | 41b4c98900445f5072be2472a9b21cb4c715d66d9b3978f1559cdbb3810d3662 |
| results/ea_historical/windows/H9/universe.parquet | 464d6713243baeac979f7a9b30636718bb29322c2f95c55db4fe4d452e9227c9 |
| results/vintage_descriptions/H10_descriptions.parquet | 6a5a0e56986ae05d36e9cb8fe0c5f7f5af9d76a58616adb32d4e937815716163 |
| results/vintage_descriptions/H11_descriptions.parquet | aa98ea5c5b78f3c4a6adb5257df2c82d939d497bd49ed259b1b97af7efaef143 |
| results/vintage_descriptions/H12_descriptions.parquet | 9308e6bd0002887dbc3a80ef6f9c813042bcc440a88d3878a54957e764de0137 |
| results/vintage_descriptions/H1_descriptions.parquet | 00edcb393c633990147cf97788c44a922e5b25de676fa37319a6b8fe9b1a74ff |
| results/vintage_descriptions/H2_descriptions.parquet | b4385d7efe43166148710ddc5d920bf39b3124157b9fef09fecc197449ec757e |
| results/vintage_descriptions/H3_descriptions.parquet | ce11098a46c3659087691e26443be1f3a1291c5539e0f3da075bf82b4704e704 |
| results/vintage_descriptions/H4_descriptions.parquet | 7997df301a3bfa48dab8c29df1419ee8262d83b469db71a5f1dad534dff75900 |
| results/vintage_descriptions/H5_descriptions.parquet | f2b8fcc7e5b9c5d7dba1971b3a50c62ba01c53909f0fbb9347da7511cfaf8ac5 |
| results/vintage_descriptions/H6_descriptions.parquet | febea2b049f408abb7b6d91574fb93b0a2941d83ff5663d72b497f3a9a3e3c2a |
| results/vintage_descriptions/H7_descriptions.parquet | 060da528cfc8d359760562d6864ed99301c6da5c586c2a55e5603cf81625548a |
| results/vintage_descriptions/H8_descriptions.parquet | fd97d79f27d42b06a1b433806789b15304b486e1dc018191f7b9d6020feb4b25 |
| results/vintage_descriptions/H9_descriptions.parquet | 1befd0a97944f8c0cdb493a032d6ad6c701410018d54451bf2c1224b6db368a0 |

### Environment

| Field | Artifact value |
|---|---|
| logical_cpu_count | 10 |
| machine | arm64 |
| numpy | 2.4.2 |
| platform | macOS-26.5.2-arm64-arm-64bit-Mach-O |
| polars | 1.39.3 |
| processor | arm |
| pytest | 9.0.2 |
| python | 3.13.2 |
| scikit-learn | 1.8.0 |
| torch | 2.11.0 |
| torch_build_config | PyTorch built with:   - GCC 4.2   - C++ Version: 201703   - clang 15.0.0   - OpenMP 201811   - LAPACK is enabled (usually provided by MKL)   - NNPACK is enabled   - CPU capability usage: DEFAULT   - Build settings: BLAS_INFO=accelerate, BUILD_TYPE=Release, COMMIT_SHA=redacted-for-anonymous-release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_PYTORCH_QNNPACK -DAT_BUILD_ARM_VEC256_WITH_SLEEF -DUSE_XNNPACK -DUSE_PYTORCH_METAL_EXPORT -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DUSE_COREML_DELEGATE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=braced-scalar-init -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wvla-extension -Wsuggest-override -Wnewline-eof -Winconsistent-missing-override -Winconsistent-missing-destructor-override -Wno-pass-failed -Wno-error=old-style-cast -Wconstant-conversion -Qunused-arguments -faligned-new -fno-math-errno -fno-trapping-math -Werror=format -DUSE_MPS -Wno-missing-braces, LAPACK_INFO=accelerate, TORCH_VERSION=2.11.0, USE_CUDA=OFF, USE_CUDNN=OFF, USE_CUSPARSELT=OFF, USE_EIGEN_FOR_BLAS=ON, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, USE_XCCL=OFF, USE_XPU=OFF,   |
| torch_debug_build | no |
| torch_deterministic_algorithms | yes |
| torch_git_version | redacted-for-anonymous-release |
| torch_num_interop_threads | 10 |
| torch_num_threads | 8 |
| torch_parallel_info | ATen/Parallel: 	at::get_num_threads() : 8 	at::get_num_interop_threads() : 10 OpenMP 201811 	omp_get_max_threads() : 8 MKLDNN not found std::thread::hardware_concurrency() : 10 Environment variables: 	OMP_NUM_THREADS : [not set] ATen parallel backend: OpenMP  |

## Mandatory self-tests

| Test | Status | Artifact detail |
|---|---|---|
| S1 | PASS | feature_orders=37740; aligned_fixture_orders=800; pre_gate_ndcg_checks=24; no_H8_H13_access |
| S2 | PASS | synthetic head learnability suite, including second-step H-A gradient |
| S3 | PASS | modern subtree request rejected before filesystem open |
| S4 | PASS | generator decimal-literal grep=0; artifact substitution suite=PASS |

## Completed fit roster

| Field | Artifact value |
|---|---|
| development_completed | 48 |
| development_expected | 48 |
| gate_completed | 18 |
| gate_expected | 18 |
| total_completed | 66 |
| total_expected | 66 |
| transition_level_optimizer_updates | 13,440 |

## Development selection

### Locked winners

| Architecture | Target softness | Application | Learning rate | Development mean NDCG@5 | Advantage over frozen blend |
|---|---|---|---|---|---|
| Global: H-B | lw | global | 0.03 | 0.412530618632 | -0.00987978858163 |
| H-A | lw | rank_band | 0.03 | 0.403654696598 | -0.0187557106148 |
| H-B | lw | global | 0.03 | 0.412530618632 | -0.00987978858163 |

### Development fit ledger

| Architecture | Target softness | Application | Learning rate | Seed | H5->H6 | H6->H7 | Equal-transition mean | Advantage over frozen blend | Final epoch loss |
|---|---|---|---|---|---|---|---|---|---|
| H-A | raw_scale | global | 0.003 | 11 | 0.382530378269 | 0.390274123073 | 0.386402250671 | -0.0360081565422 | 6.91171558135 |
| H-A | raw_scale | global | 0.003 | 29 | 0.382035342051 | 0.390035103489 | 0.38603522277 | -0.0363751844432 | 6.91155858325 |
| H-A | raw_scale | global | 0.003 | 47 | 0.381429054976 | 0.389559476649 | 0.385494265812 | -0.0369161414009 | 6.91273548923 |
| H-A | raw_scale | global | 0.03 | 11 | 0.384909237086 | 0.392147946859 | 0.388528591973 | -0.0338818152404 | 6.91315317105 |
| H-A | raw_scale | global | 0.03 | 29 | 0.384429882652 | 0.392591139614 | 0.388510511133 | -0.0338998960802 | 6.91557276559 |
| H-A | raw_scale | global | 0.03 | 47 | 0.38444904162 | 0.392066708654 | 0.388257875137 | -0.0341525320761 | 6.91003152647 |
| H-A | raw_scale | rank_band | 0.003 | 11 | 0.394429766797 | 0.411441842999 | 0.402935804898 | -0.019474602315 | 3.77203156536 |
| H-A | raw_scale | rank_band | 0.003 | 29 | 0.394594603018 | 0.412217865276 | 0.403406234147 | -0.019004173066 | 3.77151530049 |
| H-A | raw_scale | rank_band | 0.003 | 47 | 0.395608852809 | 0.411637998893 | 0.403623425851 | -0.0187869813619 | 3.77119127386 |
| H-A | raw_scale | rank_band | 0.03 | 11 | 0.394187690435 | 0.411167579476 | 0.402677634956 | -0.0197327722576 | 3.76829916021 |
| H-A | raw_scale | rank_band | 0.03 | 29 | 0.394946646414 | 0.411593304192 | 0.403269975303 | -0.0191404319098 | 3.76855491133 |
| H-A | raw_scale | rank_band | 0.03 | 47 | 0.395412203291 | 0.411425629204 | 0.403418916248 | -0.0189914909656 | 3.7682422234 |
| H-A | lw | global | 0.003 | 11 | 0.380736546484 | 0.390770442011 | 0.385753494247 | -0.0366569129658 | 7.20346685008 |
| H-A | lw | global | 0.003 | 29 | 0.379098734009 | 0.386762773661 | 0.382930753835 | -0.0394796533784 | 7.20051413862 |
| H-A | lw | global | 0.003 | 47 | 0.368818161099 | 0.371692506435 | 0.370255333767 | -0.0521550734461 | 7.19938172512 |
| H-A | lw | global | 0.03 | 11 | 0.385767096613 | 0.395418393564 | 0.390592745089 | -0.0318176621246 | 7.17396216843 |
| H-A | lw | global | 0.03 | 29 | 0.387884849001 | 0.396780477955 | 0.392332663478 | -0.0300777437352 | 7.17353894643 |
| H-A | lw | global | 0.03 | 47 | 0.34856259371 | 0.357296472384 | 0.352929533047 | -0.0694808741661 | 7.17500404322 |
| H-A | lw | rank_band | 0.003 | 11 | 0.394319798538 | 0.410759328347 | 0.402539563442 | -0.0198708437708 | 3.86648512385 |
| H-A | lw | rank_band | 0.003 | 29 | 0.395831903867 | 0.411521227247 | 0.403676565557 | -0.0187338416563 | 3.86625855762 |
| H-A | lw | rank_band | 0.003 | 47 | 0.394678632451 | 0.410057490999 | 0.402368061725 | -0.0200423454885 | 3.86628942275 |
| H-A | lw | rank_band | 0.03 | 11 | 0.395533599216 | 0.411488700287 | 0.403511149752 | -0.0188992574616 | 3.8654304619 |
| H-A | lw | rank_band | 0.03 | 29 | 0.39554102383 | 0.411615049498 | 0.403578036664 | -0.018832370549 | 3.86560848664 |
| H-A | lw | rank_band | 0.03 | 47 | 0.395559270519 | 0.41219053624 | 0.403874903379 | -0.0185355038338 | 3.86542269331 |
| H-B | raw_scale | global | 0.003 | 11 | 0.381853100902 | 0.389760547778 | 0.38580682434 | -0.0366035828732 | 7.55057273366 |
| H-B | raw_scale | global | 0.003 | 29 | 0.380022890032 | 0.389200062684 | 0.384611476358 | -0.0377989308548 | 7.54872896162 |
| H-B | raw_scale | global | 0.003 | 47 | 0.380066107765 | 0.389152716093 | 0.384609411929 | -0.0378009952842 | 7.54913308443 |
| H-B | raw_scale | global | 0.03 | 11 | 0.398419191514 | 0.412066527592 | 0.405242859553 | -0.0171675476603 | 7.04902980337 |
| H-B | raw_scale | global | 0.03 | 29 | 0.400999236444 | 0.415952766026 | 0.408476001235 | -0.0139344059779 | 7.00128993548 |
| H-B | raw_scale | global | 0.03 | 47 | 0.397813388297 | 0.410787826284 | 0.404300607291 | -0.0181097999223 | 7.01933189793 |
| H-B | raw_scale | rank_band | 0.003 | 11 | 0.381710766259 | 0.389734503162 | 0.385722634711 | -0.0366877725027 | 3.84364407621 |
| H-B | raw_scale | rank_band | 0.003 | 29 | 0.381710766259 | 0.389734503162 | 0.385722634711 | -0.0366877725027 | 3.8436344845 |
| H-B | raw_scale | rank_band | 0.003 | 47 | 0.381710766259 | 0.389734503162 | 0.385722634711 | -0.0366877725027 | 3.84367178574 |
| H-B | raw_scale | rank_band | 0.03 | 11 | 0.390901629095 | 0.406284888733 | 0.398593258914 | -0.0238171482991 | 3.78498574922 |
| H-B | raw_scale | rank_band | 0.03 | 29 | 0.390759222313 | 0.405985588875 | 0.398372405594 | -0.0240380016192 | 3.78493754282 |
| H-B | raw_scale | rank_band | 0.03 | 47 | 0.390913156953 | 0.406138584965 | 0.398525870959 | -0.0238845362539 | 3.78493848681 |
| H-B | lw | global | 0.003 | 11 | 0.381847077914 | 0.389750073396 | 0.385798575655 | -0.036611831558 | 8.25634034515 |
| H-B | lw | global | 0.003 | 29 | 0.381847077914 | 0.389761148764 | 0.385804113339 | -0.0366062938738 | 8.25631208807 |
| H-B | lw | global | 0.003 | 47 | 0.381847077914 | 0.389761148764 | 0.385804113339 | -0.0366062938738 | 8.25679857409 |
| H-B | lw | global | 0.03 | 11 | 0.399423453853 | 0.419655596504 | 0.409539525179 | -0.0128708820346 | 7.54535115941 |
| H-B | lw | global | 0.03 | 29 | 0.403927733541 | 0.424287835932 | 0.414107784736 | -0.00830262247701 | 7.25895620236 |
| H-B | lw | global | 0.03 | 47 | 0.403618942769 | 0.424270149191 | 0.41394454598 | -0.0084658612333 | 7.25877603309 |
| H-B | lw | rank_band | 0.003 | 11 | 0.381857108827 | 0.389633877035 | 0.385745492931 | -0.0366649142819 | 3.97033066072 |
| H-B | lw | rank_band | 0.003 | 29 | 0.381857108827 | 0.389633877035 | 0.385745492931 | -0.0366649142819 | 3.97032514475 |
| H-B | lw | rank_band | 0.003 | 47 | 0.381857108827 | 0.389633877035 | 0.385745492931 | -0.0366649142819 | 3.97035585193 |
| H-B | lw | rank_band | 0.03 | 11 | 0.394704747745 | 0.410984755208 | 0.402844751476 | -0.0195656557367 | 3.88801226717 |
| H-B | lw | rank_band | 0.03 | 29 | 0.394621351679 | 0.411147044648 | 0.402884198163 | -0.01952620905 | 3.88797608807 |
| H-B | lw | rank_band | 0.03 | 47 | 0.394694892504 | 0.410943742472 | 0.402819317488 | -0.0195910897252 | 3.88798453513 |

## Historical continuation gate

### Decision

| Field | Artifact value |
|---|---|
| decision | STOP |
| global_winner | H-B/lw/global |
| threshold.minimum_mean_advantage | 0.005 |
| threshold.minimum_strict_wins | 4 |
| observed.mean_advantage | -0.0258031616107 |
| observed.slot_count | 6 |
| observed.strict_wins | 0 |

### Six-slot historical table

| Transition | N | Return | TF-IDF | Frozen blend | H-A | H-A seed SD | H-B | H-B seed SD | Global winner | Winner seed SD | Winner minus blend |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H7->H8 | 1,416 | 0.427604567821 | 0.352700610646 | 0.462786883671 | 0.449376836396 | 9.02845853997e-05 | 0.429976550859 | 7.41793753332e-05 | 0.429976550859 | 7.41793753332e-05 | -0.0328103328127 |
| H8->H9 | 1,501 | 0.413774490166 | 0.318962243188 | 0.432101042959 | 0.427727986552 | 0.000518920500148 | 0.406141003177 | 0.000180168985489 | 0.406141003177 | 0.000180168985489 | -0.0259600397826 |
| H9->H10 | 1,512 | 0.380733060155 | 0.325541593379 | 0.41261308877 | 0.394260246221 | 0.0035167609435 | 0.389832428545 | 0.000255931306194 | 0.389832428545 | 0.000255931306194 | -0.0227806602253 |
| H10->H11 | 1,518 | 0.360212035684 | 0.316938494778 | 0.396883422686 | 0.374914752978 | 0.00197894370611 | 0.371823848615 | 0.000273215096549 | 0.371823848615 | 0.000273215096549 | -0.0250595740704 |
| H11->H12 | 1,534 | 0.355061750602 | 0.308085270875 | 0.389708486766 | 0.37127227407 | 0.00143131044007 | 0.365061151312 | 0.000991937768056 | 0.365061151312 | 0.000991937768056 | -0.024647335454 |
| H12->H13 | 1,545 | 0.350001738184 | 0.283116968327 | 0.362878228694 | 0.362830086878 | 0.000620064441514 | 0.339317201374 | 0.00110262306375 | 0.339317201374 | 0.00110262306375 | -0.0235610273195 |

### Equal-transition summary

| Field | Artifact value |
|---|---|
| frozen_blend_ndcg_at_5 | 0.409495192258 |
| global_winner_ndcg_at_5 | 0.383692030647 |
| global_winner_seed_dispersion | 0.000479675932561 |
| h_a_ndcg_at_5 | 0.396730363849 |
| h_b_ndcg_at_5 | 0.383692030647 |
| return_only_ndcg_at_5 | 0.381231273769 |
| tfidf_only_ndcg_at_5 | 0.317557530199 |
| winner_minus_frozen_blend | -0.0258031616107 |

## H-A interpretability

| Fold | Seed | Target softness | Application | Learning rate | Lambda | Correction mean | Correction SD | Correction p10 | Correction p50 | Correction p90 | Maximum absolute correction | Positive fraction | Score derivative wrt text cosine | Semantic-ablation NDCG change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 11 | lw | rank_band | 0.03 | 0.510709822178 | 0.0013534377848 | 0.0763582605625 | -0.1022036165 | 0.0201886272989 | 0.0774761211127 | 0.34913918376 | 0.615249428426 | 0.22237778455 | 0.0181249128485 |
| 1 | 29 | lw | rank_band | 0.03 | -0.503731071949 | -0.00139344226432 | 0.0773714397679 | -0.106573637575 | 0.0178176027257 | 0.0757225655019 | 0.353510037065 | 0.601049571002 | 0.218630731106 | 0.0184314957742 |
| 1 | 47 | lw | rank_band | 0.03 | -0.483787238598 | 0.0103145586279 | 0.0764389863978 | -0.0926977079362 | 0.0294001689181 | 0.0856478564441 | 0.344442114234 | 0.662020556603 | 0.228105068207 | 0.0184083078405 |
| 2 | 11 | lw | rank_band | 0.03 | 0.561986029148 | 0.00656521427154 | 0.0735290956779 | -0.0931965641677 | 0.0245892396197 | 0.0805171024054 | 0.368599280715 | 0.636611271009 | 0.204625494778 | 0.0136108424929 |
| 2 | 29 | lw | rank_band | 0.03 | -0.52285182476 | 0.00753380017517 | 0.0774409475148 | -0.097586299479 | 0.0268152640201 | 0.0845556747168 | 0.382012411952 | 0.644262075988 | 0.220449797809 | 0.0180771299514 |
| 2 | 47 | lw | rank_band | 0.03 | -0.545908689499 | 0.0284489257345 | 0.0771578796362 | -0.0749089520425 | 0.0476668858901 | 0.10477713868 | 0.375569596887 | 0.730353700915 | 0.223188690841 | 0.0176942163131 |
| 3 | 11 | lw | rank_band | 0.03 | 0.51846987009 | 0.075885965167 | 0.0665897738677 | -0.0139104259666 | 0.0914571546018 | 0.143623024225 | 0.299443364143 | 0.875587738692 | 0.188211850822 | 0.0139203505726 |
| 3 | 29 | lw | rank_band | 0.03 | -0.605499088764 | 0.0313064835685 | 0.0704801438253 | -0.0643631339073 | 0.0477452808991 | 0.102907241136 | 0.347085341811 | 0.745488673491 | 0.22140122205 | 0.0163311978773 |
| 3 | 47 | lw | rank_band | 0.03 | -0.513330638409 | 0.0137989212708 | 0.0675850768724 | -0.0785420257598 | 0.0293821301311 | 0.0831750441343 | 0.343360483646 | 0.673845149112 | 0.197334848344 | 0.0144680500812 |

## H-B interpretability

| Fold | Seed | Target softness | Application | Learning rate | w | b | Active gamma mean | Active gamma SD | Active gamma p10 | Active gamma p50 | Active gamma p90 | Top-five gamma mean | Top-five gamma SD | Top-five gamma p10 | Top-five gamma p50 | Top-five gamma p90 | Association: query length | Association: candidate length | Fraction gamma below zero | Fraction absolute gamma near zero | Fraction gamma above half | Semantic-ablation NDCG change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 11 | lw | global | 0.03 | -1.11990356445, 4.66065168381, 0.0166419111192 | 0.87189078331 | 0.791458773611 | 0.243498067546 | 0.85611268878 | 0.868593603373 | 0.868593603373 | 0.791549547267 | 0.243516652468 | 0.856551554799 | 0.868593603373 | 0.868593603373 | 0.851641843827 | 0.00626997533309 | 0.0705921518838 | 0.00474719011399 | 0.910615774041 | -0.00232543531385 |
| 1 | 29 | lw | global | 0.03 | -1.08961105347, 4.64320516586, 0.0175703950226 | 0.885537326336 | 0.792289576092 | 0.243605463262 | 0.856341868639 | 0.869462072849 | 0.869462072849 | 0.79238881003 | 0.243625922596 | 0.856811624765 | 0.869462072849 | 0.869462072849 | 0.851980382361 | 0.00661835172685 | 0.070580420929 | 0.00541733005956 | 0.910617328561 | -0.00256096734831 |
| 1 | 47 | lw | global | 0.03 | -1.08269536495, 4.64339971542, 0.0158806312829 | 0.88323122263 | 0.79206161307 | 0.243552827885 | 0.857346236706 | 0.869174212217 | 0.869174212217 | 0.792146156099 | 0.243569943161 | 0.857771804929 | 0.869174212217 | 0.869174212217 | 0.851945332298 | 0.00591933371761 | 0.0705799767807 | 0.00541785662287 | 0.910617772709 | -0.00243530568332 |
| 2 | 11 | lw | global | 0.03 | -1.06131839752, 4.47028112411, 0.0219733864069 | 1.21870434284 | 0.820043169358 | 0.257982868311 | 0.737642586231 | 0.904860466719 | 0.904860466719 | 0.820130842237 | 0.257997658034 | 0.736820116639 | 0.904860466719 | 0.904860466719 | 0.857531747021 | 0.00573771347142 | 0.0716778382839 | 0.00893824155524 | 0.906890502821 | 0.00966966697865 |
| 2 | 29 | lw | global | 0.03 | -1.07224071026, 4.65005159378, 0.0230568516999 | 1.1400860548 | 0.817203840814 | 0.25862945386 | 0.721235573292 | 0.902530908585 | 0.902530908585 | 0.817301486041 | 0.258648663032 | 0.720307552814 | 0.902530908585 | 0.902530908585 | 0.854433007735 | 0.00619720587032 | 0.0742905336518 | 0.00815964303772 | 0.906502494339 | 0.0102889836075 |
| 2 | 47 | lw | global | 0.03 | -1.06810986996, 4.6467795372, 0.0229699499905 | 1.1454167366 | 0.817494277603 | 0.258667224524 | 0.722174435854 | 0.902819693089 | 0.902819693089 | 0.817590428642 | 0.258685991301 | 0.721262562275 | 0.902819693089 | 0.902819693089 | 0.854541003899 | 0.00614938159725 | 0.0742864013412 | 0.00817099406364 | 0.906511239848 | 0.0102350793224 |
| 3 | 11 | lw | global | 0.03 | -1.08420884609, 4.45251655579, 0.0198190286756 | 1.36852049828 | 0.81712844191 | 0.257785027944 | 0.795945167542 | 0.901485204697 | 0.901485204697 | 0.817210205904 | 0.257810308362 | 0.794222074747 | 0.901485204697 | 0.901485204697 | 0.864845088047 | 0.00539217426471 | 0.0740647816286 | 0.00616620185479 | 0.906081890993 | 0.000771971575504 |
| 3 | 29 | lw | global | 0.03 | -1.12964344025, 4.44822120667, 0.0282399114221 | 1.35647416115 | 0.81594357896 | 0.257506330497 | 0.793907821178 | 0.90042835474 | 0.90042835474 | 0.816119601364 | 0.257555836783 | 0.793190100789 | 0.90042835474 | 0.90042835474 | 0.864648484658 | 0.00817457598023 | 0.0740758378272 | 0.00615893658843 | 0.90605495337 | 0.00106142183273 |
| 3 | 47 | lw | global | 0.03 | -1.12497758865, 4.107234478, 0.0265978164971 | 1.51623642445 | 0.822214405683 | 0.256335340404 | 0.81516996026 | 0.905749678612 | 0.905749678612 | 0.822354584845 | 0.256380642896 | 0.813022616506 | 0.905749678612 | 0.905749678612 | 0.871013480521 | 0.0072623047461 | 0.0711427025962 | 0.00651309522588 | 0.907082962473 | -0.00129685870977 |

Window-context association after slot/seed aggregation: -0.637216731221

## Retrieval and application diagnostics

| Transition | Method | Top-five overlap with return | Top-five overlap with blend | Queries differing from return | Promoted raw-rank p10 | Promoted raw-rank median | Promoted raw-rank p90 | Promoted ranks six through ten | Promoted ranks eleven through twenty | Promoted ranks twenty-one through fifty | Promoted ranks above fifty | Mean signed query-to-peer correlation | Fraction positive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H7->H8 | return-only | 1 | 0.330225988701 | 0 | undefined | undefined | undefined | 0 | 0 | 0 | 0 | 0.0575974796447 | 0.838983050847 |
| H7->H8 | frozen blend | 0.330225988701 | 1 | 0.996468926554 | 7 | 16 | 39 | 0.292914382117 | 0.338886545761 | 0.331716575285 | 0.0364824968368 | 0.0636305304486 | 0.880649717514 |
| H7->H8 | H-A | 0.677024482109 | 0.475517890772 | 0.939736346516 | 6 | 11.5 | 33.3333333333 | 0.453149928116 | 0.297075513608 | 0.249774558276 | 0 | 0.062398460401 | 0.866290018832 |
| H7->H8 | H-B | 0.310687382298 | 0.350235404896 | 0.926553672316 | 9 | 60.6666666667 | 478.466666667 | 0.126152641344 | 0.137012444973 | 0.195751760143 | 0.54108315354 | 0.0572029471743 | 0.873116760829 |
| H7->H8 | global winner (H-B) | 0.310687382298 | 0.350235404896 | 0.926553672316 | 9 | 60.6666666667 | 478.466666667 | 0.126152641344 | 0.137012444973 | 0.195751760143 | 0.54108315354 | 0.0572029471743 | 0.873116760829 |
| H8->H9 | return-only | 1 | 0.332578281146 | 0 | undefined | undefined | undefined | 0 | 0 | 0 | 0 | 0.0477847539822 | 0.80546302465 |
| H8->H9 | frozen blend | 0.332578281146 | 1 | 0.99200532978 | 7 | 16 | 39 | 0.288880015971 | 0.333599520862 | 0.340986224795 | 0.0365342383709 | 0.0504441672263 | 0.858760826116 |
| H8->H9 | H-A | 0.709349322674 | 0.468043526538 | 0.907395069953 | 6 | 11 | 31.3333333333 | 0.478230371837 | 0.300689351268 | 0.221080276895 | 0 | 0.0504999615913 | 0.837885853875 |
| H8->H9 | H-B | 0.370686209194 | 0.395158783034 | 0.923384410393 | 8 | 38.6666666667 | 364 | 0.173972239118 | 0.173406908046 | 0.216317672988 | 0.436303179848 | 0.0469220033002 | 0.84787919165 |
| H8->H9 | global winner (H-B) | 0.370686209194 | 0.395158783034 | 0.923384410393 | 8 | 38.6666666667 | 364 | 0.173972239118 | 0.173406908046 | 0.216317672988 | 0.436303179848 | 0.0469220033002 | 0.84787919165 |
| H9->H10 | return-only | 1 | 0.321693121693 | 0 | undefined | undefined | undefined | 0 | 0 | 0 | 0 | 0.0425751880642 | 0.800925925926 |
| H9->H10 | frozen blend | 0.321693121693 | 1 | 0.996031746032 | 7 | 16 | 40 | 0.275351014041 | 0.334828393136 | 0.351989079563 | 0.0378315132605 | 0.0469086480366 | 0.853835978836 |
| H9->H10 | H-A | 0.699867724868 | 0.437125220459 | 0.926587301587 | 6 | 12 | 31 | 0.470785894928 | 0.29264598118 | 0.236568123892 | 0 | 0.0453384105395 | 0.845458553792 |
| H9->H10 | H-B | 0.303306878307 | 0.333024691358 | 0.923059964727 | 9 | 68.3333333333 | 537.333333333 | 0.127148458123 | 0.141134394388 | 0.173723733181 | 0.557993414308 | 0.0430514605039 | 0.852954144621 |
| H9->H10 | global winner (H-B) | 0.303306878307 | 0.333024691358 | 0.923059964727 | 9 | 68.3333333333 | 537.333333333 | 0.127148458123 | 0.141134394388 | 0.173723733181 | 0.557993414308 | 0.0430514605039 | 0.852954144621 |
| H10->H11 | return-only | 1 | 0.307246376812 | 0 | undefined | undefined | undefined | 0 | 0 | 0 | 0 | 0.0278927235057 | 0.795783926219 |
| H10->H11 | frozen blend | 0.307246376812 | 1 | 0.998682476943 | 7 | 17 | 39 | 0.280334728033 | 0.326359832636 | 0.353936858121 | 0.0393685812096 | 0.0313593261271 | 0.83465085639 |
| H10->H11 | H-A | 0.687307861221 | 0.436319718928 | 0.929512516469 | 6 | 11.6666666667 | 31.1 | 0.477104889778 | 0.288647677219 | 0.234247433003 | 0 | 0.0298738568099 | 0.819718928415 |
| H10->H11 | H-B | 0.264909969258 | 0.297891963109 | 0.925340360123 | 10 | 92 | 668.2 | 0.107183194464 | 0.107241520387 | 0.166450213834 | 0.619125071316 | 0.0285502906761 | 0.848045674133 |
| H10->H11 | global winner (H-B) | 0.264909969258 | 0.297891963109 | 0.925340360123 | 10 | 92 | 668.2 | 0.107183194464 | 0.107241520387 | 0.166450213834 | 0.619125071316 | 0.0285502906761 | 0.848045674133 |
| H11->H12 | return-only | 1 | 0.301303780965 | 0 | undefined | undefined | undefined | 0 | 0 | 0 | 0 | 0.0372760884046 | 0.785528031291 |
| H11->H12 | frozen blend | 0.301303780965 | 1 | 0.998044328553 | 7 | 17 | 42 | 0.264228400821 | 0.320582198171 | 0.371151334204 | 0.0440380668035 | 0.0421264455827 | 0.826597131682 |
| H11->H12 | H-A | 0.721077792264 | 0.443285528031 | 0.93481095176 | 6 | 9.66666666667 | 25.6666666667 | 0.565301045112 | 0.272550707746 | 0.162148247142 | 0 | 0.040555259312 | 0.813342025206 |
| H11->H12 | H-B | 0.298870056497 | 0.345893089961 | 0.924380704042 | 9.33333333333 | 66.8333333333 | 529.566666667 | 0.123000075397 | 0.125229735969 | 0.187698724378 | 0.564071464256 | 0.0391304726871 | 0.840286831812 |
| H11->H12 | global winner (H-B) | 0.298870056497 | 0.345893089961 | 0.924380704042 | 9.33333333333 | 66.8333333333 | 529.566666667 | 0.123000075397 | 0.125229735969 | 0.187698724378 | 0.564071464256 | 0.0391304726871 | 0.840286831812 |
| H12->H13 | return-only | 1 | 0.291521035599 | 0 | undefined | undefined | undefined | 0 | 0 | 0 | 0 | 0.0279537987573 | 0.777346278317 |
| H12->H13 | frozen blend | 0.291521035599 | 1 | 0.998705501618 | 7 | 17 | 41 | 0.261099945185 | 0.32852183446 | 0.367988306231 | 0.0423899141239 | 0.0291032368136 | 0.814886731392 |
| H12->H13 | H-A | 0.709212513484 | 0.431758360302 | 0.920172599784 | 6 | 10 | 26.3333333333 | 0.556276966927 | 0.276576211036 | 0.167146822037 | 0 | 0.0298178191449 | 0.804099244876 |
| H12->H13 | H-B | 0.262955771305 | 0.290097087379 | 0.925134843581 | 10.3333333333 | 100.666666667 | 704.9 | 0.0997186250736 | 0.102994811957 | 0.16428773896 | 0.632998824009 | 0.0264862857513 | 0.819201725998 |
| H12->H13 | global winner (H-B) | 0.262955771305 | 0.290097087379 | 0.925134843581 | 10.3333333333 | 100.666666667 | 704.9 | 0.0997186250736 | 0.102994811957 | 0.16428773896 | 0.632998824009 | 0.0264862857513 | 0.819201725998 |

## Target-softness audit

The LW target rescales the raw correlation target by `c_t`. The last
column gives the raw-scale temperature that has the same softness.

| Transition | LW shrinkage | Target scale c_t | Fixed target temperature | Raw-equivalent target temperature |
|---|---|---|---|---|
| H1->H2 | 0.751095235348 | 0.248904764652 | 0.1 | 0.401760087396 |
| H2->H3 | 0.735408842564 | 0.264591157436 | 0.1 | 0.377941579639 |
| H3->H4 | 0.760586500168 | 0.239413499832 | 0.1 | 0.417687390519 |
| H4->H5 | 0.747152388096 | 0.252847611904 | 0.1 | 0.3954951334 |
| H5->H6 | 0.752402365208 | 0.247597634792 | 0.1 | 0.403881079413 |
| H6->H7 | 0.730084598064 | 0.269915401936 | 0.1 | 0.37048645347 |
| H7->H8 | 0.710107445717 | 0.289892554283 | 0.1 | 0.344955393033 |
| H8->H9 | 0.74048012495 | 0.25951987505 | 0.1 | 0.385326942612 |
| H9->H10 | 0.747424244881 | 0.252575755119 | 0.1 | 0.395920819687 |
| H10->H11 | 0.813909053802 | 0.186090946198 | 0.1 | 0.537371656404 |
| H11->H12 | 0.750870347023 | 0.249129652977 | 0.1 | 0.401397420199 |
| H12->H13 | 0.804657518864 | 0.195342481136 | 0.1 | 0.511921418313 |

## Context audit

| Transition | Length p10 | Length p50 | Length p90 | Fraction at collection cap | Missing lengths | Window length | Training-scaler mean | Training-scaler population SD |
|---|---|---|---|---|---|---|---|---|
| H7->H8 | 8000 | 8000 | 8000 | 0.908192090395 | 0 | 82 | 8.92028746789 | 0.275209792277 |
| H8->H9 | 8000 | 8000 | 8000 | 0.900066622252 | 0 | 85 | 8.92028746789 | 0.275209792277 |
| H9->H10 | 7422.7 | 8000 | 8000 | 0.893518518519 | 0 | 84 | 8.91853519398 | 0.277271544699 |
| H10->H11 | 7307.2 | 8000 | 8000 | 0.892621870883 | 0 | 81 | 8.91853519398 | 0.277271544699 |
| H11->H12 | 7498.3 | 8000 | 8000 | 0.895697522816 | 0 | 86 | 8.91670209344 | 0.279020136457 |
| H12->H13 | 7601.4 | 8000 | 8000 | 0.89644012945 | 0 | 83 | 8.91670209344 | 0.279020136457 |

## Computational audit

| Field | Artifact value |
|---|---|
| run_wall_time_seconds | 583.680883 |
| peak_resident_memory_mb | 2340.1875 |
| trainable_parameters.H-A | 114 |
| trainable_parameters.H-B | 4 |

### Full-transition scoring calls

| Transition | Method | Wall time seconds |
|---|---|---|
| H5->H6 | development/dev-ha-raw_scale-global-lr0p003-seed11 | 0.0682743750003 |
| H6->H7 | development/dev-ha-raw_scale-global-lr0p003-seed11 | 0.0672408330065 |
| H5->H6 | development/dev-ha-raw_scale-global-lr0p003-seed29 | 0.0551756249988 |
| H6->H7 | development/dev-ha-raw_scale-global-lr0p003-seed29 | 0.058530125003 |
| H5->H6 | development/dev-ha-raw_scale-global-lr0p003-seed47 | 0.0583083340025 |
| H6->H7 | development/dev-ha-raw_scale-global-lr0p003-seed47 | 0.0570612909942 |
| H5->H6 | development/dev-ha-raw_scale-global-lr0p03-seed11 | 0.0660027080012 |
| H6->H7 | development/dev-ha-raw_scale-global-lr0p03-seed11 | 0.0625017920029 |
| H5->H6 | development/dev-ha-raw_scale-global-lr0p03-seed29 | 0.0565864999953 |
| H6->H7 | development/dev-ha-raw_scale-global-lr0p03-seed29 | 0.0571626250021 |
| H5->H6 | development/dev-ha-raw_scale-global-lr0p03-seed47 | 0.0568785839932 |
| H6->H7 | development/dev-ha-raw_scale-global-lr0p03-seed47 | 0.0569867090016 |
| H5->H6 | development/dev-ha-raw_scale-rank_band-lr0p003-seed11 | 0.154823165998 |
| H6->H7 | development/dev-ha-raw_scale-rank_band-lr0p003-seed11 | 0.155567499998 |
| H5->H6 | development/dev-ha-raw_scale-rank_band-lr0p003-seed29 | 0.158396875006 |
| H6->H7 | development/dev-ha-raw_scale-rank_band-lr0p003-seed29 | 0.153733915999 |
| H5->H6 | development/dev-ha-raw_scale-rank_band-lr0p003-seed47 | 0.158776124998 |
| H6->H7 | development/dev-ha-raw_scale-rank_band-lr0p003-seed47 | 0.154071334 |
| H5->H6 | development/dev-ha-raw_scale-rank_band-lr0p03-seed11 | 0.176762875002 |
| H6->H7 | development/dev-ha-raw_scale-rank_band-lr0p03-seed11 | 0.168890292 |
| H5->H6 | development/dev-ha-raw_scale-rank_band-lr0p03-seed29 | 0.155782417001 |
| H6->H7 | development/dev-ha-raw_scale-rank_band-lr0p03-seed29 | 0.152373750003 |
| H5->H6 | development/dev-ha-raw_scale-rank_band-lr0p03-seed47 | 0.154476749995 |
| H6->H7 | development/dev-ha-raw_scale-rank_band-lr0p03-seed47 | 0.151738832996 |
| H5->H6 | development/dev-ha-lw-global-lr0p003-seed11 | 0.0585396250026 |
| H6->H7 | development/dev-ha-lw-global-lr0p003-seed11 | 0.0560164999988 |
| H5->H6 | development/dev-ha-lw-global-lr0p003-seed29 | 0.0593462079996 |
| H6->H7 | development/dev-ha-lw-global-lr0p003-seed29 | 0.0578258329988 |
| H5->H6 | development/dev-ha-lw-global-lr0p003-seed47 | 0.0585809580007 |
| H6->H7 | development/dev-ha-lw-global-lr0p003-seed47 | 0.0600324170009 |
| H5->H6 | development/dev-ha-lw-global-lr0p03-seed11 | 0.0596086250007 |
| H6->H7 | development/dev-ha-lw-global-lr0p03-seed11 | 0.0580402080013 |
| H5->H6 | development/dev-ha-lw-global-lr0p03-seed29 | 0.0639467080036 |
| H6->H7 | development/dev-ha-lw-global-lr0p03-seed29 | 0.0600257080005 |
| H5->H6 | development/dev-ha-lw-global-lr0p03-seed47 | 0.0588639580019 |
| H6->H7 | development/dev-ha-lw-global-lr0p03-seed47 | 0.0575287079992 |
| H5->H6 | development/dev-ha-lw-rank_band-lr0p003-seed11 | 0.161046875 |
| H6->H7 | development/dev-ha-lw-rank_band-lr0p003-seed11 | 0.157579415994 |
| H5->H6 | development/dev-ha-lw-rank_band-lr0p003-seed29 | 0.162758084007 |
| H6->H7 | development/dev-ha-lw-rank_band-lr0p003-seed29 | 0.154692124997 |
| H5->H6 | development/dev-ha-lw-rank_band-lr0p003-seed47 | 0.1587 |
| H6->H7 | development/dev-ha-lw-rank_band-lr0p003-seed47 | 0.153423207994 |
| H5->H6 | development/dev-ha-lw-rank_band-lr0p03-seed11 | 0.161506457996 |
| H6->H7 | development/dev-ha-lw-rank_band-lr0p03-seed11 | 0.155159833004 |
| H5->H6 | development/dev-ha-lw-rank_band-lr0p03-seed29 | 0.158698667001 |
| H6->H7 | development/dev-ha-lw-rank_band-lr0p03-seed29 | 0.152632792 |
| H5->H6 | development/dev-ha-lw-rank_band-lr0p03-seed47 | 0.156024457996 |
| H6->H7 | development/dev-ha-lw-rank_band-lr0p03-seed47 | 0.155426749996 |
| H5->H6 | development/dev-hb-raw_scale-global-lr0p003-seed11 | 0.0435546670051 |
| H6->H7 | development/dev-hb-raw_scale-global-lr0p003-seed11 | 0.0447934999975 |
| H5->H6 | development/dev-hb-raw_scale-global-lr0p003-seed29 | 0.035854624999 |
| H6->H7 | development/dev-hb-raw_scale-global-lr0p003-seed29 | 0.0359912909989 |
| H5->H6 | development/dev-hb-raw_scale-global-lr0p003-seed47 | 0.0362659579987 |
| H6->H7 | development/dev-hb-raw_scale-global-lr0p003-seed47 | 0.0365485830043 |
| H5->H6 | development/dev-hb-raw_scale-global-lr0p03-seed11 | 0.0360124590006 |
| H6->H7 | development/dev-hb-raw_scale-global-lr0p03-seed11 | 0.036537333006 |
| H5->H6 | development/dev-hb-raw_scale-global-lr0p03-seed29 | 0.0360182500008 |
| H6->H7 | development/dev-hb-raw_scale-global-lr0p03-seed29 | 0.036743208002 |
| H5->H6 | development/dev-hb-raw_scale-global-lr0p03-seed47 | 0.0437430419988 |
| H6->H7 | development/dev-hb-raw_scale-global-lr0p03-seed47 | 0.0426818749984 |
| H5->H6 | development/dev-hb-raw_scale-rank_band-lr0p003-seed11 | 0.134116834 |
| H6->H7 | development/dev-hb-raw_scale-rank_band-lr0p003-seed11 | 0.133076833998 |
| H5->H6 | development/dev-hb-raw_scale-rank_band-lr0p003-seed29 | 0.134768791999 |
| H6->H7 | development/dev-hb-raw_scale-rank_band-lr0p003-seed29 | 0.132616082999 |
| H5->H6 | development/dev-hb-raw_scale-rank_band-lr0p003-seed47 | 0.136313084004 |
| H6->H7 | development/dev-hb-raw_scale-rank_band-lr0p003-seed47 | 0.139555417001 |
| H5->H6 | development/dev-hb-raw_scale-rank_band-lr0p03-seed11 | 0.136386625003 |
| H6->H7 | development/dev-hb-raw_scale-rank_band-lr0p03-seed11 | 0.135284083997 |
| H5->H6 | development/dev-hb-raw_scale-rank_band-lr0p03-seed29 | 0.134716917004 |
| H6->H7 | development/dev-hb-raw_scale-rank_band-lr0p03-seed29 | 0.132072167005 |
| H5->H6 | development/dev-hb-raw_scale-rank_band-lr0p03-seed47 | 0.136888084002 |
| H6->H7 | development/dev-hb-raw_scale-rank_band-lr0p03-seed47 | 0.131838583002 |
| H5->H6 | development/dev-hb-lw-global-lr0p003-seed11 | 0.0362689579997 |
| H6->H7 | development/dev-hb-lw-global-lr0p003-seed11 | 0.0365440419992 |
| H5->H6 | development/dev-hb-lw-global-lr0p003-seed29 | 0.035719583997 |
| H6->H7 | development/dev-hb-lw-global-lr0p003-seed29 | 0.0374628749996 |
| H5->H6 | development/dev-hb-lw-global-lr0p003-seed47 | 0.0359532919974 |
| H6->H7 | development/dev-hb-lw-global-lr0p003-seed47 | 0.0385990420036 |
| H5->H6 | development/dev-hb-lw-global-lr0p03-seed11 | 0.0360957080047 |
| H6->H7 | development/dev-hb-lw-global-lr0p03-seed11 | 0.0365097079994 |
| H5->H6 | development/dev-hb-lw-global-lr0p03-seed29 | 0.0365469999961 |
| H6->H7 | development/dev-hb-lw-global-lr0p03-seed29 | 0.0357771250056 |
| H5->H6 | development/dev-hb-lw-global-lr0p03-seed47 | 0.0358211249986 |
| H6->H7 | development/dev-hb-lw-global-lr0p03-seed47 | 0.0359001249963 |
| H5->H6 | development/dev-hb-lw-rank_band-lr0p003-seed11 | 0.134203708003 |
| H6->H7 | development/dev-hb-lw-rank_band-lr0p003-seed11 | 0.132494666002 |
| H5->H6 | development/dev-hb-lw-rank_band-lr0p003-seed29 | 0.135029332996 |
| H6->H7 | development/dev-hb-lw-rank_band-lr0p003-seed29 | 0.137309540994 |
| H5->H6 | development/dev-hb-lw-rank_band-lr0p003-seed47 | 0.135558375005 |
| H6->H7 | development/dev-hb-lw-rank_band-lr0p003-seed47 | 0.136160333001 |
| H5->H6 | development/dev-hb-lw-rank_band-lr0p03-seed11 | 0.149703625 |
| H6->H7 | development/dev-hb-lw-rank_band-lr0p03-seed11 | 0.150400209001 |
| H5->H6 | development/dev-hb-lw-rank_band-lr0p03-seed29 | 0.135209707994 |
| H6->H7 | development/dev-hb-lw-rank_band-lr0p03-seed29 | 0.135668124996 |
| H5->H6 | development/dev-hb-lw-rank_band-lr0p03-seed47 | 0.134908375003 |
| H6->H7 | development/dev-hb-lw-rank_band-lr0p03-seed47 | 0.132518165999 |
| H7->H8 | H-A/seed11 | 0.161735832997 |
| H7->H8 | H-A/seed11/semantic-ablation | 0.154999041006 |
| H8->H9 | H-A/seed11 | 0.181801084 |
| H8->H9 | H-A/seed11/semantic-ablation | 0.170105875004 |
| H7->H8 | H-A/seed29 | 0.160093874998 |
| H7->H8 | H-A/seed29/semantic-ablation | 0.154859082999 |
| H8->H9 | H-A/seed29 | 0.171634833001 |
| H8->H9 | H-A/seed29/semantic-ablation | 0.183432624995 |
| H7->H8 | H-A/seed47 | 0.162483166001 |
| H7->H8 | H-A/seed47/semantic-ablation | 0.153081542005 |
| H8->H9 | H-A/seed47 | 0.171660958003 |
| H8->H9 | H-A/seed47/semantic-ablation | 0.169998833997 |
| H9->H10 | H-A/seed11 | 0.179595916001 |
| H9->H10 | H-A/seed11/semantic-ablation | 0.176440208001 |
| H10->H11 | H-A/seed11 | 0.177305375 |
| H10->H11 | H-A/seed11/semantic-ablation | 0.174322625004 |
| H9->H10 | H-A/seed29 | 0.176988832995 |
| H9->H10 | H-A/seed29/semantic-ablation | 0.171983959 |
| H10->H11 | H-A/seed29 | 0.175427458998 |
| H10->H11 | H-A/seed29/semantic-ablation | 0.175048708996 |
| H9->H10 | H-A/seed47 | 0.180162084005 |
| H9->H10 | H-A/seed47/semantic-ablation | 0.172623416998 |
| H10->H11 | H-A/seed47 | 0.176082500002 |
| H10->H11 | H-A/seed47/semantic-ablation | 0.174292249998 |
| H11->H12 | H-A/seed11 | 0.193182667004 |
| H11->H12 | H-A/seed11/semantic-ablation | 0.180433499998 |
| H12->H13 | H-A/seed11 | 0.199144165999 |
| H12->H13 | H-A/seed11/semantic-ablation | 0.185917250004 |
| H11->H12 | H-A/seed29 | 0.189383334 |
| H11->H12 | H-A/seed29/semantic-ablation | 0.184821041999 |
| H12->H13 | H-A/seed29 | 0.189123625001 |
| H12->H13 | H-A/seed29/semantic-ablation | 0.182537875 |
| H11->H12 | H-A/seed47 | 0.191327540997 |
| H11->H12 | H-A/seed47/semantic-ablation | 0.179534499999 |
| H12->H13 | H-A/seed47 | 0.193681249999 |
| H12->H13 | H-A/seed47/semantic-ablation | 0.185298457996 |
| H7->H8 | H-B/seed11 | 0.038165083999 |
| H7->H8 | H-B/seed11/semantic-ablation | 0.037946041004 |
| H8->H9 | H-B/seed11 | 0.0408675419967 |
| H8->H9 | H-B/seed11/semantic-ablation | 0.0438448330024 |
| H7->H8 | H-B/seed29 | 0.037039791001 |
| H7->H8 | H-B/seed29/semantic-ablation | 0.0375064999971 |
| H8->H9 | H-B/seed29 | 0.040797833004 |
| H8->H9 | H-B/seed29/semantic-ablation | 0.0401923329991 |
| H7->H8 | H-B/seed47 | 0.0372775840005 |
| H7->H8 | H-B/seed47/semantic-ablation | 0.0369753749983 |
| H8->H9 | H-B/seed47 | 0.0415445420003 |
| H8->H9 | H-B/seed47/semantic-ablation | 0.0403142919968 |
| H9->H10 | H-B/seed11 | 0.041222916996 |
| H9->H10 | H-B/seed11/semantic-ablation | 0.040680457998 |
| H10->H11 | H-B/seed11 | 0.0415844999952 |
| H10->H11 | H-B/seed11/semantic-ablation | 0.0406237079951 |
| H9->H10 | H-B/seed29 | 0.0419889580007 |
| H9->H10 | H-B/seed29/semantic-ablation | 0.0410227920001 |
| H10->H11 | H-B/seed29 | 0.0425784169929 |
| H10->H11 | H-B/seed29/semantic-ablation | 0.0424476669941 |
| H9->H10 | H-B/seed47 | 0.0416401250041 |
| H9->H10 | H-B/seed47/semantic-ablation | 0.0411542910006 |
| H10->H11 | H-B/seed47 | 0.0423613329986 |
| H10->H11 | H-B/seed47/semantic-ablation | 0.0446729999967 |
| H11->H12 | H-B/seed11 | 0.0448734579986 |
| H11->H12 | H-B/seed11/semantic-ablation | 0.0415070419986 |
| H12->H13 | H-B/seed11 | 0.0548852499996 |
| H12->H13 | H-B/seed11/semantic-ablation | 0.0548193330033 |
| H11->H12 | H-B/seed29 | 0.0434319169945 |
| H11->H12 | H-B/seed29/semantic-ablation | 0.0415228330021 |
| H12->H13 | H-B/seed29 | 0.0472718339952 |
| H12->H13 | H-B/seed29/semantic-ablation | 0.0463719170002 |
| H11->H12 | H-B/seed47 | 0.0420865000051 |
| H11->H12 | H-B/seed47/semantic-ablation | 0.0420382500015 |
| H12->H13 | H-B/seed47 | 0.0451590000011 |
| H12->H13 | H-B/seed47/semantic-ablation | 0.0456669170017 |

## Scope and interpretation

The experiment measures historical next-window residual-peer retrieval.
It does not measure valuation accuracy, analyst forecasts, portfolio profit,
transaction costs, or a causal effect. Context associations and pairwise
diagnostics are descriptive and do not alter selection or the gate.
