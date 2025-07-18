import { ArrowDownOutlined, PlusOutlined } from "@ant-design/icons";
import { useEffect, useState } from "react";
import { Space } from "antd";

import { useAxiosPrivate } from "../../../hooks/useAxiosPrivate";
import { useAlertStore } from "../../../store/alert-store";
import { useSessionStore } from "../../../store/session-store";
import { CustomButton } from "../../widgets/custom-button/CustomButton";
import { AddCustomToolFormModal } from "../add-custom-tool-form-modal/AddCustomToolFormModal";
import { ViewTools } from "../view-tools/ViewTools";
import "./ListOfTools.css";
import { useExceptionHandler } from "../../../hooks/useExceptionHandler";
import { ToolNavBar } from "../../navigations/tool-nav-bar/ToolNavBar";
import { SharePermission } from "../../widgets/share-permission/SharePermission";
import usePostHogEvents from "../../../hooks/usePostHogEvents.js";
import { ImportTool } from "../import-tool/ImportTool";

function ListOfTools() {
  const [isListLoading, setIsListLoading] = useState(false);
  const [openAddTool, setOpenAddTool] = useState(false);
  const [openImportTool, setOpenImportTool] = useState(false);
  const [isImportLoading, setIsImportLoading] = useState(false);
  const [editItem, setEditItem] = useState(null);
  const { sessionDetails } = useSessionStore();
  const { setPostHogCustomEvent } = usePostHogEvents();

  const { setAlertDetails } = useAlertStore();
  const axiosPrivate = useAxiosPrivate();

  const [listOfTools, setListOfTools] = useState([]);
  const [filteredListOfTools, setFilteredListOfTools] = useState([]);
  const handleException = useExceptionHandler();
  const [isEdit, setIsEdit] = useState(false);
  const [promptDetails, setPromptDetails] = useState(null);
  const [openSharePermissionModal, setOpenSharePermissionModal] =
    useState(false);
  const [isPermissionEdit, setIsPermissionEdit] = useState(false);
  const [isShareLoading, setIsShareLoading] = useState(false);
  const [allUserList, setAllUserList] = useState([]);

  useEffect(() => {
    getListOfTools();
  }, []);

  useEffect(() => {
    setFilteredListOfTools(listOfTools);
  }, [listOfTools]);

  const getListOfTools = () => {
    const requestOptions = {
      method: "GET",
      url: `/api/v1/unstract/${sessionDetails?.orgId}/prompt-studio/ `,
      headers: {
        "X-CSRFToken": sessionDetails?.csrfToken,
      },
    };

    setIsListLoading(true);
    axiosPrivate(requestOptions)
      .then((res) => {
        const data = res?.data;
        setListOfTools(data);
        setFilteredListOfTools(data);
      })
      .catch((err) => {
        setAlertDetails(
          handleException(err, "Failed to get the list of tools")
        );
      })
      .finally(() => {
        setIsListLoading(false);
      });
  };

  const handleAddNewTool = (body) => {
    let method = "POST";
    let url = `/api/v1/unstract/${sessionDetails?.orgId}/prompt-studio/`;
    const isEdit = editItem && Object.keys(editItem)?.length > 0;
    if (isEdit) {
      method = "PATCH";
      url += `${editItem?.tool_id}/`;
    }
    return new Promise((resolve, reject) => {
      const requestOptions = {
        method,
        url,
        headers: {
          "X-CSRFToken": sessionDetails?.csrfToken,
          "Content-Type": "application/json",
        },
        data: body,
      };

      axiosPrivate(requestOptions)
        .then((res) => {
          const tool = res?.data;
          updateList(isEdit, tool);
          setOpenAddTool(false);
          resolve(res?.data);
        })
        .catch((err) => {
          reject(err);
        });
    });
  };

  const updateList = (isEdit, data) => {
    let tools = [...listOfTools];

    if (isEdit) {
      tools = tools.map((item) =>
        item?.tool_id === data?.tool_id ? data : item
      );
      setEditItem(null);
    } else {
      tools.push(data);
    }
    setListOfTools(tools);
  };

  const handleEdit = (_event, tool) => {
    const editToolData = [...listOfTools].find(
      (item) => item?.tool_id === tool.tool_id
    );
    if (!editToolData) {
      return;
    }
    setIsEdit(true);
    setEditItem(editToolData);
    setOpenAddTool(true);
  };

  const handleDelete = (_event, tool) => {
    const requestOptions = {
      method: "DELETE",
      url: `/api/v1/unstract/${sessionDetails?.orgId}/prompt-studio/${tool.tool_id}`,
      headers: {
        "X-CSRFToken": sessionDetails?.csrfToken,
      },
    };

    axiosPrivate(requestOptions)
      .then(() => {
        const tools = [...listOfTools].filter(
          (filterToll) => filterToll?.tool_id !== tool.tool_id
        );
        setListOfTools(tools);
        setAlertDetails({
          type: "success",
          content: `${tool?.tool_name} - Deleted successfully`,
        });
      })
      .catch((err) => {
        setAlertDetails(handleException(err, "Failed to Delete"));
      });
  };

  const onSearch = (search, setSearch) => {
    if (search?.length === 0) {
      setSearch(listOfTools);
    }
    const filteredList = [...listOfTools].filter((tool) => {
      const name = tool.tool_name?.toUpperCase();
      const searchUpperCase = search.toUpperCase();
      return name.includes(searchUpperCase);
    });
    setSearch(filteredList);
  };

  const showAddTool = () => {
    setEditItem(null);
    setIsEdit(false);
    setOpenAddTool(true);
  };

  const handleNewProjectBtnClick = () => {
    showAddTool();

    try {
      setPostHogCustomEvent("intent_new_ps_project", {
        info: "Clicked on '+ New Project' button",
      });
    } catch (err) {
      // If an error occurs while setting custom posthog event, ignore it and continue
    }
  };

  const handleImportProject = (file, selectedAdapters) => {
    try {
      setPostHogCustomEvent("intent_tool_import_project", {
        info: "Importing project from projects list",
        file_name: file.name,
      });
    } catch (err) {
      // If an error occurs while setting custom posthog event, ignore it and continue
    }

    setIsImportLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    // Add selected adapter IDs to the form data
    if (selectedAdapters) {
      formData.append("llm_adapter_id", selectedAdapters.llm);
      formData.append("vector_db_adapter_id", selectedAdapters.vectorDb);
      formData.append("embedding_adapter_id", selectedAdapters.embedding);
      formData.append("x2text_adapter_id", selectedAdapters.x2text);
    }

    const requestOptions = {
      method: "POST",
      url: `/api/v1/unstract/${sessionDetails?.orgId}/prompt-studio/project-transfer/`,
      headers: {
        "X-CSRFToken": sessionDetails?.csrfToken,
      },
      data: formData,
    };

    axiosPrivate(requestOptions)
      .then((response) => {
        const {
          message,
          warning,
          needs_adapter_config: needsAdapterConfig,
        } = response.data;

        setAlertDetails({
          type: needsAdapterConfig ? "warning" : "success",
          content: warning ? `${message} ${warning}` : message,
        });
        setOpenImportTool(false);

        // Refresh the list of tools to show the new imported project
        getListOfTools();
      })
      .catch((err) => {
        setAlertDetails(handleException(err, "Failed to import project"));
      })
      .finally(() => {
        setIsImportLoading(false);
      });
  };

  const CustomButtons = () => {
    return (
      <Space gap={16}>
        <CustomButton
          type="default"
          icon={<ArrowDownOutlined />}
          onClick={() => setOpenImportTool(true)}
          loading={isImportLoading}
        >
          Import Project
        </CustomButton>
        <CustomButton
          type="primary"
          icon={<PlusOutlined />}
          onClick={handleNewProjectBtnClick}
        >
          New Project
        </CustomButton>
      </Space>
    );
  };

  const handleShare = (_event, promptProject, isEdit) => {
    const requestOptions = {
      method: "GET",
      url: `/api/v1/unstract/${sessionDetails?.orgId}/prompt-studio/users/${promptProject?.tool_id}`,
      headers: {
        "X-CSRFToken": sessionDetails?.csrfToken,
      },
    };
    setIsShareLoading(true);
    getAllUsers();
    axiosPrivate(requestOptions)
      .then((res) => {
        setOpenSharePermissionModal(true);
        setPromptDetails(res?.data);
        setIsPermissionEdit(isEdit);
      })
      .catch((err) => {
        setAlertDetails(handleException(err));
      })
      .finally(() => {
        setIsShareLoading(false);
      });
  };

  const getAllUsers = () => {
    setIsShareLoading(true);
    const requestOptions = {
      method: "GET",
      url: `/api/v1/unstract/${sessionDetails?.orgId}/users/`,
    };

    axiosPrivate(requestOptions)
      .then((response) => {
        const users = response?.data?.members || [];
        setAllUserList(
          users.map((user) => ({
            id: user?.id,
            email: user?.email,
          }))
        );
      })
      .catch((err) => {
        setAlertDetails(handleException(err, "Failed to load"));
      })
      .finally(() => {
        setIsShareLoading(false);
      });
  };

  const onShare = (userIds, adapter) => {
    const requestOptions = {
      method: "PATCH",
      url: `/api/v1/unstract/${sessionDetails?.orgId}/prompt-studio/${adapter?.tool_id}`,
      headers: {
        "X-CSRFToken": sessionDetails?.csrfToken,
      },
      data: { shared_users: userIds },
    };
    axiosPrivate(requestOptions)
      .then((response) => {
        setOpenSharePermissionModal(false);
      })
      .catch((err) => {
        setAlertDetails(handleException(err, "Failed to load"));
      });
  };

  return (
    <>
      <ToolNavBar
        title={"Prompt Studio"}
        enableSearch
        onSearch={onSearch}
        searchList={listOfTools}
        setSearchList={setFilteredListOfTools}
        CustomButtons={CustomButtons}
      />
      <div className="list-of-tools-layout">
        <div className="list-of-tools-island">
          <div className="list-of-tools-body">
            <ViewTools
              isLoading={isListLoading}
              isEmpty={!listOfTools?.length}
              listOfTools={filteredListOfTools}
              setOpenAddTool={setOpenAddTool}
              handleEdit={handleEdit}
              handleDelete={handleDelete}
              titleProp="tool_name"
              descriptionProp="description"
              iconProp="icon"
              idProp="tool_id"
              type="Prompt Project"
              handleShare={handleShare}
            />
          </div>
        </div>
      </div>
      {openAddTool && (
        <AddCustomToolFormModal
          open={openAddTool}
          setOpen={setOpenAddTool}
          editItem={editItem}
          isEdit={isEdit}
          handleAddNewTool={handleAddNewTool}
        />
      )}
      <ImportTool
        open={openImportTool}
        setOpen={setOpenImportTool}
        onImport={handleImportProject}
        loading={isImportLoading}
      />
      <SharePermission
        open={openSharePermissionModal}
        setOpen={setOpenSharePermissionModal}
        adapter={promptDetails}
        permissionEdit={isPermissionEdit}
        loading={isShareLoading}
        allUsers={allUserList}
        onApply={onShare}
      />
    </>
  );
}

export { ListOfTools };
