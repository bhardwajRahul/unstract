import { Col, Row } from "antd";
import { useState, useEffect } from "react";

import { useAxiosPrivate } from "../../../hooks/useAxiosPrivate";
import { useExceptionHandler } from "../../../hooks/useExceptionHandler";
import { useAlertStore } from "../../../store/alert-store";
import { useCustomToolStore } from "../../../store/custom-tool-store";
import { useSessionStore } from "../../../store/session-store";
import { DocumentManager } from "../document-manager/DocumentManager";
import { Header } from "../header/Header";
import { SettingsModal } from "../settings-modal/SettingsModal";
import { ToolsMain } from "../tools-main/ToolsMain";
import "./ToolIde.css";
import usePostHogEvents from "../../../hooks/usePostHogEvents.js";
import { PageTitle } from "../../widgets/page-title/PageTitle.jsx";

let PromptShareModal;
let PromptShareLink;
let CloneTitle;
let HeaderPublic;

try {
  PromptShareModal =
    require("../../../plugins/prompt-studio-public-share/public-share-modal/PromptShareModal.jsx").PromptShareModal;
  PromptShareLink =
    require("../../../plugins/prompt-studio-public-share/public-link-modal/PromptShareLink.jsx").PromptShareLink;
  HeaderPublic =
    require("../../../plugins/prompt-studio-public-share/header-public/HeaderPublic.jsx").HeaderPublic;
} catch (err) {
  // Do nothing if plugins are not loaded.
}
try {
  CloneTitle =
    require("../../../plugins/prompt-studio-clone/clone-title-modal/CloneTitle.jsx").CloneTitle;
} catch (err) {
  // Do nothing if plugins are not loaded.
}
function ToolIde() {
  const [openSettings, setOpenSettings] = useState(false);
  const {
    details,
    updateCustomTool,
    isMultiPassExtractLoading,
    selectedDoc,
    indexDocs,
    pushIndexDoc,
    deleteIndexDoc,
    shareId,
    isPublicSource,
  } = useCustomToolStore();
  const { sessionDetails } = useSessionStore();
  const { setAlertDetails } = useAlertStore();
  const axiosPrivate = useAxiosPrivate();
  const handleException = useExceptionHandler();
  const { setPostHogCustomEvent } = usePostHogEvents();
  const [openShareLink, setOpenShareLink] = useState(false);
  const [openShareConfirmation, setOpenShareConfirmation] = useState(false);
  const [openShareModal, setOpenShareModal] = useState(false);
  const [openCloneModal, setOpenCloneModal] = useState(false);

  useEffect(() => {
    if (openShareModal) {
      if (shareId) {
        setOpenShareConfirmation(false);
        setOpenShareLink(true);
      } else {
        setOpenShareConfirmation(true);
        setOpenShareLink(false);
      }
    }
  }, [shareId, openShareModal]);

  useEffect(() => {
    if (!openShareModal) {
      setOpenShareConfirmation(false);
      setOpenShareLink(false);
    }
  }, [openShareModal]);

  const generateIndex = async (doc) => {
    const docId = doc?.document_id;

    if (indexDocs.includes(docId)) {
      setAlertDetails({
        type: "error",
        content: "This document is already getting indexed",
      });
      return;
    }

    const body = {
      document_id: docId,
    };

    const requestOptions = {
      method: "POST",
      url: `/api/v1/unstract/${sessionDetails?.orgId}/prompt-studio/index-document/${details?.tool_id}`,
      headers: {
        "X-CSRFToken": sessionDetails?.csrfToken,
        "Content-Type": "application/json",
      },
      data: body,
    };

    pushIndexDoc(docId);
    return axiosPrivate(requestOptions)
      .then(() => {
        setAlertDetails({
          type: "success",
          content: `${doc?.document_name} - Indexed successfully`,
        });

        try {
          setPostHogCustomEvent("intent_success_ps_indexed_file", {
            info: "Indexing completed",
          });
        } catch (err) {
          // If an error occurs while setting custom posthog event, ignore it and continue
        }
      })
      .catch((err) => {
        setAlertDetails(
          handleException(err, `${doc?.document_name} - Failed to index`)
        );
      })
      .finally(() => {
        deleteIndexDoc(docId);
      });
  };

  const handleUpdateTool = async (body) => {
    const requestOptions = {
      method: "PATCH",
      url: `/api/v1/unstract/${sessionDetails?.orgId}/prompt-studio/${details?.tool_id}/`,
      headers: {
        "X-CSRFToken": sessionDetails?.csrfToken,
        "Content-Type": "application/json",
      },
      data: body,
    };

    return axiosPrivate(requestOptions)
      .then((res) => {
        return res;
      })
      .catch((err) => {
        throw err;
      });
  };

  const handleDocChange = (doc) => {
    if (isMultiPassExtractLoading) {
      setAlertDetails({
        type: "error",
        content: "Please wait for the run to complete",
      });
      return;
    }

    const prevSelectedDoc = selectedDoc;
    const data = {
      selectedDoc: doc,
    };
    updateCustomTool(data);
    if (isPublicSource) {
      return;
    }
    const body = {
      output: doc?.document_id,
    };
    handleUpdateTool(body)
      .then((res) => {
        const updatedToolData = res?.data;
        updateCustomTool({ details: updatedToolData });
      })
      .catch((err) => {
        const revertSelectedDoc = {
          selectedDoc: prevSelectedDoc,
        };
        updateCustomTool(revertSelectedDoc);
        setAlertDetails(handleException(err, "Failed to select the document"));
      });
  };

  return (
    <div className="tool-ide-layout">
      <PageTitle title={details?.tool_name} />
      {isPublicSource && HeaderPublic && <HeaderPublic />}
      <div>
        <Header
          handleUpdateTool={handleUpdateTool}
          setOpenSettings={setOpenSettings}
          setOpenShareModal={setOpenShareModal}
          setOpenCloneModal={setOpenCloneModal}
        />
      </div>
      <div
        className={isPublicSource ? "public-tool-ide-body" : "tool-ide-body"}
      >
        <div className="tool-ide-body-2">
          <Row className="tool-ide-main">
            <Col span={12} className="tool-ide-col">
              <div className="tool-ide-prompts">
                <ToolsMain />
              </div>
            </Col>
            <Col span={12} className="tool-ide-col">
              <div className="tool-ide-pdf">
                <DocumentManager
                  generateIndex={generateIndex}
                  handleUpdateTool={handleUpdateTool}
                  handleDocChange={handleDocChange}
                />
              </div>
            </Col>
          </Row>
        </div>
      </div>
      <div className="height-50" />
      <SettingsModal
        open={openSettings}
        setOpen={setOpenSettings}
        handleUpdateTool={handleUpdateTool}
      />
      {PromptShareModal && (
        <PromptShareModal
          open={openShareConfirmation}
          setOpenShareModal={setOpenShareModal}
          setOpenShareConfirmation={setOpenShareConfirmation}
        />
      )}
      {PromptShareLink && (
        <PromptShareLink
          open={openShareLink}
          setOpenShareModal={setOpenShareModal}
          setOpenShareLink={setOpenShareLink}
        />
      )}
      {CloneTitle && (
        <CloneTitle
          open={openCloneModal}
          setOpenCloneModal={setOpenCloneModal}
        />
      )}
    </div>
  );
}

export { ToolIde };
